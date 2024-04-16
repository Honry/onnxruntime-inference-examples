import { AutoProcessor, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.15.1';

//'@xenova/transformers';
import { get_new_tokens } from './generation_utils.js';
import { attention_mask_update, cache_update } from './post_processing.js';
import { log, getModelOPFS, convertToFloat32Array, convertToUint16Array } from './utils.js';

// wrapper around onnxruntime and model
export class Whisper {
    constructor(url, provider, dataType) {
        this.url = url;
        this.provider = provider;
        this.dataType = dataType;
        ort.env.wasm.simd = true;

        this.models = {
            'encoder': { url: 'whisper_base_encoder_lm.onnx', sess: null },
            'decoder': { url: 'whisper_base_decoder_static_non_kvcache_lm.onnx', sess: null },
            'decoder_cached': { url: 'whisper_base_decoder_static_kvcache_128_lm.onnx', sess: null },
        };

        this.max_sequence_length = 128;
        // No. of tokens to be used for decoder 1st inference
        this.num_init_tokens = 4;
        // Whisper was trained using 16000 Hz as sampling rate, fixing this value for dataset preparation
        this.sampling_rate = 16000;
        this.processor = null;
        this.tokenizer = null;
        this.pad_session = null;
        this.scatternd_session = null;
        this.indices = new ort.Tensor('int64', new BigInt64Array(24), [1, 8, 1, 3]); // used for replace kv cache
    }

    async create_whisper_processor() {
        // processor contains feature extractor
        this.processor = await AutoProcessor.from_pretrained('openai/whisper-base');
    }

    async create_whisper_tokenizer() {
        // processor contains feature extractor
        this.tokenizer = await AutoTokenizer.from_pretrained('openai/whisper-base', { config: { do_normalize: true } });
    }

    async create_ort_sessions() {
        const options = {
            executionProviders: [{
                name: this.provider,
                deviceType: 'gpu',
            }],
        };

        try {
            this.pad_session = await ort.InferenceSession.create('/models/pad.onnx');
            this.scatternd_session = await ort.InferenceSession.create('/models/scatternd.onnx');
        } catch (e) {
            log(`Error: ${e}`);
        }
        for (let name of Object.keys(this.models)) {
            try {
                let url = this.url + this.models[name]['url'];
                if (this.dataType == 'float16') {
                    url = url.replace('.onnx', '_fp16_layernorm.onnx');
                } else {
                    url = url.replace('.onnx', '_layernorm.onnx');
                }
                const modelBuffer = await getModelOPFS(`${name}_${this.dataType}`, url, false);
                this.models[name]['sess'] = await ort.InferenceSession.create(modelBuffer, options);
                log(`Model ${url} loaded`);
            } catch (e) {
                log(`Error: ${e}`);
            }
        };
    }

    async run(audio_data) {
        // -----------------------------------FEATURE EXTRACTION-----------------------------------------
        // const audio = await read_audio('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac', 16000);
        // const audio = await read_audio(audio_data, sampling_rate);
        let start = performance.now();
        const { input_features } = await this.processor(audio_data);
        // -----------------------------------ENCODER INFERENCE-----------------------------------------
        // run encoder to get output
        let encoder_inputs = {
            type: this.dataType,
            data: input_features.data,
            dims: input_features.dims,
        };
        if (this.dataType == 'float16') {
            encoder_inputs.data = convertToUint16Array(encoder_inputs.data);
        }
        console.log(`  pre-processing time: ${(performance.now() - start).toFixed(2)} ms`);
        start = performance.now();
        const { last_hidden_state } = await this.models['encoder']['sess'].run({
            input_features: new ort.Tensor(encoder_inputs.type, encoder_inputs.data, encoder_inputs.dims)
        });
        console.log(`  encoder inference time: ${(performance.now() - start).toFixed(2)} ms`);
        start = performance.now();
        // -----------------------------------DECODER 1ST INFERENCE-----------------------------------------
        // create list of tokens for english language and transcribe task, no need of time stamps
        // TODO: CHANGE FROM HARDCODED VALUES
        let tokens = [50258, 50259, 50359, 50363];
        // let tokens = [50258, 50259, 50359, 50364]; // keep timestep token
        const attention_mask = [1, 1, 1, 1];
        // create decoder input for the first inference
        const decoder_input = {
            'input_ids': new ort.Tensor('int32', new Int32Array(tokens), [1, 4]),
            'attention_mask': new ort.Tensor('int32', new Int32Array(attention_mask), [1, 4]),
            'encoder_hidden_states': last_hidden_state,
        };

        console.log(`  non-kv cache decoder input preparation time: ${(performance.now() - start).toFixed(2)} ms`);
        start = performance.now();
        // run the first inference which generates SA and CA KV cache
        const decoder_output = await this.models['decoder']['sess'].run(decoder_input);
        console.log(`  non-kv cache decoder inference time: ${(performance.now() - start).toFixed(2)} ms`);
        start = performance.now();
        let logits = decoder_output['logits']['cpuData'];

        if (this.dataType == 'float16') {
            logits = convertToFloat32Array(logits);
        }
        // find out the token with highest probability, cast INT64 to INT32
        const new_token = get_new_tokens(logits, [1, 4, 51865]);

        // add token to final buffer
        tokens = tokens.concat(new_token);

        // for 2+ inference, we don't need encoder hidden states as input to OV model
        delete decoder_input.encoder_hidden_states;

        // -----------------------------------DECODER 2 INFERENCE-----------------------------------------
        // prepare inputs for decoder kv cache

        // create 1x1 array for input_ids
        decoder_input['input_ids'] = new ort.Tensor('int32', new Int32Array([new_token]), [1, 1]);

        // pad attention mask to max_seq_length
        decoder_input['attention_mask'] = new ort.Tensor('int64', attention_mask_update(new BigInt64Array([1n, 1n, 1n, 1n]),
            0, this.max_sequence_length, this.num_init_tokens), [1, 128]);
        // create position_ids as input, value should be same of No. of prefill tokens
        decoder_input['position_ids'] = new ort.Tensor('int32', new Int32Array([this.num_init_tokens]), [1]);

        // fill decoder kv cache model inputs with cross attention KV cache data from decoder 1st inference
        for (let i = 0; i < 6; i++) {
            // encoder key/value
            decoder_input[`past_key_values.${i}.encoder.key`] = decoder_output[`present_key_values.${i}.encoder.key`];
            decoder_input[`past_key_values.${i}.encoder.value`] = decoder_output[`present_key_values.${i}.encoder.value`];

            // pad decoder key/value
            const key_feeds = { 'input': decoder_output[`present_key_values.${i}.decoder.key`] };
            const value_feeds = { 'input': decoder_output[`present_key_values.${i}.decoder.value`] };
            const key_results = await this.pad_session.run(key_feeds);
            const value_results = await this.pad_session.run(value_feeds);
            decoder_input[`past_key_values.${i}.decoder.key`] = key_results.output;
            decoder_input[`past_key_values.${i}.decoder.value`] = value_results.output;
        }

        // modify the self attention kv cache in place
        // cache_update(
        //     decoder_input,
        //     decoder_output,
        //     0,
        //     this.max_sequence_length,
        //     this.num_init_tokens,
        //     this.num_init_tokens,
        //     this.dataType);

        const position_ids = new Int32Array(decoder_input['position_ids'].cpuData.buffer);
        // run complete inference for every item in dataset
        for (let i = 4; i < this.max_sequence_length; i++) {
            console.log(`  decoder iteration ${i - 3}: input preparation time: ${(performance.now() - start).toFixed(2)} ms`);
            start = performance.now();
            const decoder_cached_output = await this.models['decoder_cached']['sess'].run(decoder_input);
            console.log(`  decoder iteration ${i - 3}: inference time: ${(performance.now() - start).toFixed(2)} ms`);
            start = performance.now();
            // find out the token with highest probability, cast INT64 to INT32
            let logits = decoder_cached_output['logits']['cpuData'];
            if (this.dataType == 'float16') {
                logits = convertToFloat32Array(logits);
            }
            const new_token = get_new_tokens(logits, [1, 1, 51865]);

            // add token to final buffer
            tokens = tokens.concat(new_token);
            // break if the new token is eos_token_id: 50257 (end of sequence)
            // or the max sequence length is reached
            if (new_token == 50257 || i == this.max_sequence_length - 1) {
                break;
            }
            // ----------------------------------POST PROCESSING---------------------------------------
            // the following code creates decoder input for the next inference
            decoder_input['input_ids'] = new ort.Tensor('int32', new Int32Array([new_token]), [1, 1]);

            // increment the position_ids
            position_ids[0] = position_ids[0] + 1;

            // update mask using position id
            attention_mask_update(decoder_input['attention_mask'].cpuData, i, this.max_sequence_length, this.num_init_tokens, position_ids[0]);
            // modify the kv cache in place
            // cache_update(decoder_input, decoder_cached_output, i, this.max_sequence_length, this.num_init_tokens, position_ids[0], this.dataType);


            // modify the kv cache in place using scatterND
            // fill up indices
            for (let i = 0; i < 8; i++) {
                const baseIndex = i * 3;
                this.indices.cpuData[baseIndex] = BigInt(0); // first indice，value always set to 0，as batch size is 1
                this.indices.cpuData[baseIndex + 1] = BigInt(i); // second indice，scope [0, 7]
                this.indices.cpuData[baseIndex + 2] = BigInt(position_ids[0] - 1); // third indice，depends on position
            }

            for (let i = 0; i < 6; i++) {
                const key_feeds = {
                    'data': decoder_input[`past_key_values.${i}.decoder.key`],
                    'indices': this.indices,
                    'updates': decoder_cached_output[`present_key_values.${i}.decoder.key`],
                }
                const value_feeds = {
                    'data': decoder_input[`past_key_values.${i}.decoder.value`],
                    'indices': this.indices,
                    'updates': decoder_cached_output[`present_key_values.${i}.decoder.value`],
                }
                const key_results = await this.scatternd_session.run(key_feeds);
                const value_results = await this.scatternd_session.run(value_feeds);
                decoder_input[`past_key_values.${i}.decoder.key`] = key_results.output;
                decoder_input[`past_key_values.${i}.decoder.value`] = value_results.output;
            }
        }
        // add token to sentence decode time
        const sentence = await this.tokenizer.decode(tokens, { skip_special_tokens: true });
        console.log(`  post-processing time: ${(performance.now() - start).toFixed(2)} ms`);
        return sentence;
    }
}
