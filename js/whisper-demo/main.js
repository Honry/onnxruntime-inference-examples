// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run whisper in onnxruntime-web.
//

import { Whisper } from './whisper.js';
import { log, concatBuffer, concatBufferArray } from './utils.js';
import VADBuilder, { VADMode, VADEvent } from "./vad/embedded.js";
import { lcm } from "./vad/math.js";

const kSampleRate = 16000;
const kMaxAudioLengthInSec = 30;
const kSteps = kSampleRate * kMaxAudioLengthInSec;

// whisper class
let whisper;

let provider = 'webnn';
let deviceType = 'gpu';
let dataType = 'float16';

// audio context
let context = null;
let stream;

export function getAudioContext() {
    return context;
}

// for audio capture
// This enum states the current speech state.
const SpeechStates = {
    UNINITIALIZED: 0,
    PROCESSING: 1,
    PAUSED: 2,
    FINISHED: 3,
};
let speechState = SpeechStates.UNINITIALIZED;

let streamingNode = null;
let sourceNode = null;
let audioSourceNode = null;
let streamSourceNode = null;
let audioChunks = []; // member {isSubChunk: boolean, data: Float32Array}
let subAudioChunks = [];
let accumulateSubChunks = false; // Accumulate the sub audio chunks for one processing.
let chunkLength = 1 / 25; // length in sec of one audio chunk from AudioWorklet processor, recommended by vad
let maxChunkLength = 1; // max audio length in sec for a single audio processing
let maxAudioLength = 10; // max audio length in sec for rectification, must not be greater than 30 sec
let maxUnprocessedAudioLength = 0;
let maxProcessAudioBufferLength = 0;
let verbose = false;
let silenceAudioCounter = 0;
// check if last audio processing is completed, to avoid race condition
let lastProcessingCompleted = true;
// check if last speech processing is completed when restart speech
let lastSpeechCompleted = true;

// involve webrtcvad to detect voice activity
let VAD = null;
let vad = null;

let singleAudioChunk = null; // one time audio process buffer
let subAudioChunkLength = 0; // length of a sub audio chunk
let subText = '';

let baseUrl = '.';

const blacklistTags = [
    '[inaudible]',
    ' [inaudible]',
    '[ Inaudible ]',
    '[INAUDIBLE]',
    ' [INAUDIBLE]',
    '[BLANK_AUDIO]',
    ' [BLANK_AUDIO]',
    ' [no audio]',
    '[no audio]',
    '[silent]',
];

function updateConfig(options) {
    if (options !== undefined) {
        if (options.baseUrl !== undefined) {
            baseUrl = options.baseUrl;
        }
        if (options.provider !== undefined) {
            provider = options.provider;
        }
        if (options.deviceType !== undefined) {
            deviceType = options.deviceType;
        }
        if (options.dataType !== undefined) {
            dataType = options.dataType;
        }
        if (options.maxChunkLength !== undefined) {
            maxChunkLength = options.maxChunkLength;
        }
        if (options.chunkLength !== undefined) {
            chunkLength = options.chunkLength;
        }
        if (options.maxAudioLength !== undefined) {
            maxAudioLength = Math.min(options.maxAudioLength, kMaxAudioLengthInSec);
        }
        if (options.verbose !== undefined) {
            verbose = options.verbose;
        }
        if (options.accumulateSubChunks !== undefined) {
            accumulateSubChunks = options.accumulateSubChunks;
        }
    }
}

// Initalize audio context and whisper models. The caller must pass ort, AutoProcessor and AutoTokenizer
// because they may be loaded via different ways (webpack, import or script tag).
export async function initWhisper(ort, AutoProcessor, AutoTokenizer, options) {
    updateConfig(options);
    log(`Execution provider: ${provider}`);
    log("loading model...");
    try {
        context = new AudioContext({ sampleRate: kSampleRate });
        const whisper_url = location.href.includes('github.io') ?
            'https://huggingface.co/lwanming/whisper-base-static-shape/resolve/main/' :
            `${baseUrl}/models/`;
        whisper = new Whisper(whisper_url, provider, deviceType, dataType, ort, AutoProcessor, AutoTokenizer, verbose);
        await whisper.create_whisper_processor();
        await whisper.create_whisper_tokenizer();
        await whisper.create_ort_sessions();
        log("Ready to transcribe...");
    } catch (e) {
        log(`Error: ${e}`);
        return false;
    }
    return true;
}

// process audio buffer
export async function process_audio(audio, starttime, idx, pos, textarea, progress) {
    if (idx < audio.length) {
        // not done
        try {
            // update progress bar
            progress.style.width = (idx * 100 / audio.length).toFixed(1) + "%";
            progress.textContent = progress.style.width;
            // run inference for 30 sec
            const xa = audio.slice(idx, idx + kSteps);
            const ret = await whisper.run(xa);
            // append results to textarea 
            textarea.value += ret;
            textarea.scrollTop = textarea.scrollHeight;
            await process_audio(audio, starttime, idx + kSteps, pos + kMaxAudioLengthInSec, textarea, progress);
        } catch (e) {
            log(`Error: ${e}`);
        }
    } else {
        // done with audio buffer
        const processing_time = ((performance.now() - starttime) / 1000);
        const total = (audio.length / kSampleRate);
        document.getElementById('latency').innerText = `${(total / processing_time).toFixed(1)} x realtime`;
        log(`${document.getElementById('latency').innerText}, total ${processing_time.toFixed(1)}sec processing time for ${total.toFixed(1)}sec audio`);
    }
}

let recognition;

// start speech recognition for mic or an audio source if it presents.
export async function startSpeech(recognitionClient, audio_src) {
    if (!lastSpeechCompleted) {
        log('Last speech-to-text has not completed yet, try later...');
        return false;
    }
    recognition = recognitionClient;
    recognition._onstart();
    speechState = SpeechStates.PROCESSING;
    await captureAudioStream(audio_src);
    if (streamingNode != null) {
        streamingNode.port.postMessage({ message: "STOP_PROCESSING", data: false });
    }
    maxUnprocessedAudioLength = 0;
    maxProcessAudioBufferLength = 0;
    return true;
}

// stop speech
export async function stopSpeech() {
    if (streamingNode != null) {
        streamingNode.port.postMessage({ message: "STOP_PROCESSING", data: true });
        speechState = SpeechStates.PAUSED;
    }
    silenceAudioCounter = 0;
    // push last singleAudioChunk to audioChunks, in case it is ignored.
    if (singleAudioChunk != null) {
        audioChunks.push({ 'isSubChunk': false, 'data': singleAudioChunk });
        singleAudioChunk = null;
        if (lastProcessingCompleted && lastSpeechCompleted && audioChunks.length > 0) {
            await processAudioBuffer();
        }
    }

    if (sourceNode) {
        sourceNode.disconnect();
        sourceNode = null;
    }

    if (streamingNode) {
        streamingNode.disconnect();
        streamingNode = null;
    }

    console.warn(`max process audio length: ${maxProcessAudioBufferLength} sec`);
    console.warn(`max unprocessed audio length: ${maxUnprocessedAudioLength} sec`);
    recognition._onend();
    // if (stream) {
    //     stream.getTracks().forEach(track => track.stop());
    // }
    // if (context) {
    //     // context.close().then(() => context = null);
    //     await context.suspend();
    // }
}

// use AudioWorklet API to capture real-time audio
async function captureAudioStream(audio_src) {
    try {
        if (context && context.state === 'suspended') {
            await context.resume();
        }
        // Get user's microphone and connect it to the AudioContext.
        if (!stream && !audio_src) {
            stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    autoGainControl: true,
                    noiseSuppression: true,
                    channelCount: 1,
                    latency: 0
                }
            });
        }

        VAD = await VADBuilder();
        vad = new VAD(VADMode.AGGRESSIVE, kSampleRate);

        if (audio_src && !audioSourceNode) {
            audioSourceNode = new MediaElementAudioSourceNode(context, { mediaElement: audio_src });
        }

        if (!audio_src && !streamSourceNode) {
            streamSourceNode = new MediaStreamAudioSourceNode(context, { mediaStream: stream });
        }

        if (!streamingNode) {
            await context.audioWorklet.addModule(`${baseUrl}/streaming_processor.js`);
            // 128 is the minimum length for audio worklet processing.
            const minBufferSize = vad.getMinBufferSize(lcm(chunkLength * kSampleRate, 128));
            console.log(`VAD minBufferSize: ${minBufferSize / kSampleRate} sec`);
            const streamProperties = {
                minBufferSize: minBufferSize,
            };

            streamingNode = new AudioWorkletNode(
                context,
                'streaming-processor',
                {
                    processorOptions: streamProperties,
                },
            );

            streamingNode.port.onmessage = async (e) => {
                if (e.data.message === 'START_TRANSCRIBE') {
                    const frame = VAD.floatTo16BitPCM(e.data.buffer); // VAD requires Int16Array input
                    const res = vad.processBuffer(frame);
                    // has voice
                    if (res == VADEvent.VOICE) {
                        singleAudioChunk = concatBuffer(singleAudioChunk, e.data.buffer);
                        // meet max audio chunk length for a single process, split it.
                        if (singleAudioChunk.length >= kSampleRate * maxChunkLength) {
                            if (subAudioChunkLength == 0) {
                                // subAudioChunkLength >= kSampleRate * maxChunkLength
                                subAudioChunkLength = singleAudioChunk.length;
                            }
                            audioChunks.push({ 'isSubChunk': true, 'data': singleAudioChunk });
                            singleAudioChunk = null;
                        }

                        silenceAudioCounter = 0;
                    } else { // no voice
                        silenceAudioCounter++;
                        // if only one silence chunk exists between two voice chunks,
                        // just treat it as a continous audio chunk.
                        if (singleAudioChunk != null && silenceAudioCounter > 1) {
                            audioChunks.push({ 'isSubChunk': false, 'data': singleAudioChunk });
                            singleAudioChunk = null;
                        }
                    }

                    // new audio is coming, and no audio is processing
                    if (lastProcessingCompleted && audioChunks.length > 0) {
                        await processAudioBuffer();
                    }
                }
            };
        }

        if (audio_src) {
            sourceNode = audioSourceNode;
        } else {
            sourceNode = streamSourceNode;
        }

        sourceNode
            .connect(streamingNode)
            .connect(context.destination);

        if (audio_src) {
            // Play audio source to speaker.
            sourceNode.connect(context.destination);
        }
    } catch (e) {
        recognition._onerror({
            error: 'audio-capture',
            message: 'Failed to process audio data.',
        });
        log(`Error on capturing audio: ${e}`);
    }
}

async function processAudioBuffer() {
    lastProcessingCompleted = false;
    let processBuffer;
    const audioChunk = audioChunks.shift();
    // it is sub audio chunk, need to do rectification at last sub chunk
    if (audioChunk.isSubChunk) {
        subAudioChunks.push(audioChunk.data);
        // if the speech is pause, and it is the last audio chunk, concat the subAudioChunks to do rectification
        if (speechState == SpeechStates.PAUSED && audioChunks.length == 1) {
            processBuffer = concatBufferArray(subAudioChunks);
            subAudioChunks = []; // clear subAudioChunks
        } else if (subAudioChunks.length * maxChunkLength >= maxAudioLength) {
            // if total length of subAudioChunks >= maxAudioLength sec,
            // force to break it from subAudioChunks to reduce latency
            // because it has to wait for more than 10 sec to do audio processing.
            processBuffer = concatBufferArray(subAudioChunks);
            subAudioChunks = [];
        } else {
            if (accumulateSubChunks) {
                processBuffer = concatBufferArray(subAudioChunks);
            } else {
                processBuffer = audioChunk.data;
            }
        }
    } else {
        // Slience detected, concat all subAudoChunks to do rectification
        if (subAudioChunks.length > 0) {
            subAudioChunks.push(audioChunk.data); // append sub chunk's next neighbor
            processBuffer = concatBufferArray(subAudioChunks);
            subAudioChunks = []; // clear subAudioChunks
        } else {
            // No other subAudioChunks, just process this one.
            processBuffer = audioChunk.data;
        }
    }

    // ignore too small audio chunk, e.g. 0.16 sec
    // per testing, audios less than 0.16 sec are almost blank audio
    const processBufferLength = processBuffer.length / kSampleRate;
    if (processBufferLength > 0.16) {
        const start = performance.now();
        const ret = await whisper.run(processBuffer);
        console.log(`${processBufferLength} sec audio processing time: ${((performance.now() - start) / 1000).toFixed(2)} sec`);
        if (verbose) {
            console.log('result:', ret);
        }
        // ignore slient, inaudible audio output, i.e. '[BLANK_AUDIO]'
        if (!blacklistTags.includes(ret)) {
            if (subAudioChunks.length > 0) {
                if (accumulateSubChunks) {
                    subText = ret;
                } else {
                    subText += ret;
                }
                recognition._onresult(subText, false);
            } else {
                subText = '';
                recognition._onresult(ret, true);
            }
        }
    } else {
        console.warn(`drop too small audio chunk: ${processBufferLength}`);
    }
    if (processBufferLength > maxProcessAudioBufferLength) {
        maxProcessAudioBufferLength = processBufferLength;
    }
    lastProcessingCompleted = true;
    if (audioChunks.length > 0) {
        // TODO? throttle the un-processed audio chunks?
        // In order to catch up latest audio to achieve real-time effects.
        let unprocessedAudioLength = 0;
        for (let i = 0; i < audioChunks.length; ++i) {
            unprocessedAudioLength += audioChunks[i].data.length;
        }
        unprocessedAudioLength /= kSampleRate;
        console.warn(`un-processed audio chunk length: ${(unprocessedAudioLength)} sec`);
        if (unprocessedAudioLength > maxUnprocessedAudioLength) {
            maxUnprocessedAudioLength = unprocessedAudioLength;
        }
        // recusive audioBuffer in audioChunks
        lastSpeechCompleted = false;
        await processAudioBuffer();
    } else {
        lastSpeechCompleted = true;
    }
}

export function isLastSpeechCompleted() {
    return lastSpeechCompleted;
}
