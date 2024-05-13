// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// An example how to run sd-turbo with WebNN or WebGPU in onnxruntime-web.
//

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

/*
 * get configuration from url
*/
function getConfig() {
    const query = window.location.search.substring(1);
    var config = {
        model: location.href.includes("github.io") ? "https://huggingface.co/lwanming/sd-turbo-ort-web/resolve/main" : "models",
        provider: "webnn",
        device: "gpu",
        threads: "1",
        images: "1",
    };
    let vars = query.split("&");
    for (var i = 0; i < vars.length; i++) {
        let pair = vars[i].split("=");
        if (pair[0] in config) {
            config[pair[0]] = decodeURIComponent(pair[1]);
        } else if (pair[0].length > 0) {
            throw new Error("unknown argument: " + pair[0]);
        }
    }
    config.threads = parseInt(config.threads);
    config.images = parseInt(config.images);
    return config;
}

/*
 * initialize latents with random noise
 */
function randn_latents(shape, noise_sigma) {
    function randn() {
        // Use the Box-Muller transform
        let u = Math.random();
        let v = Math.random();
        let z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        return z;
    }
    let size = 1;
    shape.forEach(element => {
        size *= element;
    });

    let data = new Float32Array(size);
    // Loop over the shape dimensions
    for (let i = 0; i < size; i++) {
        data[i] = randn() * noise_sigma;
    }
    return data;
}

/*
 * fetch and cache model
 */
async function fetchAndCache(base_url, model_path) {
    const url = `${base_url}/${model_path}`;
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
            log(`${model_path} (network)`);
        } else {
            log(`${model_path} (cached)`);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`${model_path} (network)`);
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

/*
 * load models used in the pipeline
 */
async function load_models(models) {
    log("Execution provider: " + config.provider);
    const cache = await caches.open("onnx");
    let missing = 0;
    for (const [name, model] of Object.entries(models)) {
        const url = `${config.model}/${model.url}`;
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            missing += model.size;
        }
    }
    if (missing > 0) {
        log(`downloading ${missing} MB from network ... it might take a while`);
    } else {
        log("loading...");
    }
    for (const [name, model] of Object.entries(models)) {
        try {
            const start = performance.now();
            const model_bytes = await fetchAndCache(config.model, model.url);
            const sess_opt = { ...opt, ...model.opt };
            models[name].sess = await ort.InferenceSession.create(model_bytes, sess_opt);
            const stop = performance.now();
            log(`${model.url} in ${(stop - start).toFixed(1)}ms`);
        } catch (e) {
            log(`${model.url} failed, ${e}`);
        }
    }
    log("ready.");
}

const config = getConfig();

const models = {
    "unet": {
        // original model from dwayne, then I dump new one from local graph optimization.
        url: "unet/model_layernorm.onnx", size: 640,
        opt: { graphOptimizationLevel: 'disabled' }, // avoid wasm heap issue (need Wasm memory 64)
    },
    "text_encoder": {
        // orignal model from guschmue, I convert the output to fp16.
        url: "text_encoder/model_layernorm.onnx", size: 1700,
        opt: { graphOptimizationLevel: 'disabled' }, // it is already an optimized model
        // opt: { freeDimensionOverrides: { batch_size: 1, sequence_length: 77 } },
    },
    "vae_decoder": {
        // use guschmue's model has precision lose in webnn caused by instanceNorm op,
        // covert the model to run instanceNorm in fp32 (insert cast nodes).
        url: "vae_decoder/model.onnx", size: 95,
        // opt: { freeDimensionOverrides: { batch_size: 1, num_channels_latent: 4, height_latent: 64, width_latent: 64 } }
        opt: { freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 } }

    },
    "safety_checker": {
        url: "safety_checker/model.onnx", size: 100,
        opt: { freeDimensionOverrides: { batch: 1, channels: 3, height: 224, width: 224 } }
    },
}

// ort.env.wasm.wasmPaths = 'dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

let tokenizer;
let loading;
const sigma = 14.6146;
const gamma = 0;
const vae_scaling_factor = 0.18215;
const text = document.getElementById("user-input");

text.value = "Paris with the river in the background";

const opt = {
    executionProviders: [config.provider],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
        session: {
            disable_prepacking: "1",
            use_device_allocator_for_initializers: "1",
            use_ort_model_bytes_directly: "1",
            use_ort_model_bytes_for_initializers: "1"
        }
    },
};

switch (config.provider) {
    case "webgpu":
        if (!("gpu" in navigator)) {
            throw new Error("webgpu is NOT supported");
        }
        opt.preferredOutputLocation = { last_hidden_state: "gpu-buffer" };
        break;
    case "webnn":
        if (!("ml" in navigator)) {
            throw new Error("webnn is NOT supported");
        }
        opt.executionProviders = [{
            name: "webnn",
            deviceType: config.device,
            powerPreference: 'default'
        }];
        break;
}

// Event listener for Ctrl + Enter or CMD + Enter
document.getElementById('user-input').addEventListener('keydown', function (e) {
    if (e.ctrlKey && e.key === 'Enter') {
        generate_image();
    }
});
document.getElementById('send-button').addEventListener('click', function (e) {
    generate_image()
});

/*
 * scale the latents
*/
function scale_model_inputs(t) {
    const d_i = t.data;
    const d_o = new Float32Array(d_i.length);

    const divi = (sigma ** 2 + 1) ** 0.5;
    for (let i = 0; i < d_i.length; i++) {
        d_o[i] = d_i[i] / divi;
    }
    return new ort.Tensor(d_o, t.dims);
}

/*
 * Poor mens EulerA step
 * Since this example is just sd-turbo, implement the absolute minimum needed to create an image
 * Maybe next step is to support all sd flavors and create a small helper model in onnx can deal
 * much more efficient with latents.
 */
function step(model_output, sample) {
    const d_o = new Float32Array(model_output.data.length);
    const prev_sample = new ort.Tensor(d_o, model_output.dims);
    const sigma_hat = sigma * (gamma + 1);

    for (let i = 0; i < model_output.data.length; i++) {
        const pred_original_sample = sample.data[i] - sigma_hat * model_output.data[i];
        const derivative = (sample.data[i] - pred_original_sample) / sigma_hat;
        const dt = 0 - sigma_hat;
        d_o[i] = (sample.data[i] + derivative * dt) / vae_scaling_factor;
    }
    return prev_sample;
}

/**
 * draw an image from tensor
 * @param {ort.Tensor} t
 * @param {number} image_nr
*/
function draw_image(t, image_nr) {
    let pix = t.data;
    for (var i = 0; i < pix.length; i++) {
        let x = pix[i];
        x = x / 2 + 0.5
        if (x < 0.) x = 0.;
        if (x > 1.) x = 1.;
        pix[i] = x;
    }
    const imageData = t.toImageData({ tensorLayout: 'NCHW', format: 'RGB' });
    const canvas = document.getElementById(`img_canvas_${image_nr}`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    const div = document.getElementById(`img_div_${image_nr}`);
    div.style.opacity = 1.
}

function draw_out_image(t) {
    const imageData = t.toImageData({ tensorLayout: 'NHWC', format: 'RGB' });
    const canvas = document.getElementById(`img_canvas_safety`);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
    const div = document.getElementById(`safety_img`);
    div.style.opacity = 1.
}

function resize_image(image_nr, targetWidth, targetHeight) {
    // let canvas = document.createElement('canvas');
    // Use img_canvas_test to ensure the input
    const canvas = document.getElementById(`img_canvas_test`);
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    let ctx = canvas.getContext('2d');
    let canvas_source = document.getElementById(`img_canvas_${image_nr}`);
    ctx.drawImage(canvas_source, 0, 0, canvas_source.width, canvas_source.height, 0, 0, targetWidth, targetHeight);
    const div = document.getElementById(`test_img`);
    div.style.opacity = 1.
    let imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);

    return imageData;
}

function normalizeImageData(imageData) {
    const mean = [0.48145466, 0.4578275, 0.40821073];
    const std = [0.26862954, 0.26130258, 0.27577711];
    const { data, width, height } = imageData;
    const numPixels = width * height;

    for (let i = 0; i < numPixels; i++) {
        const offset = i * 4;
        for (let c = 0; c < 3; c++) {
            const normalizedValue = ((data[offset + c] / 255) - mean[c]) / std[c];
            data[offset + c] = Math.round(normalizedValue * 255);
        }
    }

    return imageData;
}

function get_tensor_from_image(imageData, format) {
    const { data, width, height } = imageData;
    const numPixels = width * height;
    const channels = 3; 
    const rearrangedData = new Float32Array(numPixels * channels);
    let destOffset = 0;
  
    for (let i = 0; i < numPixels; i++) {
      const srcOffset = i * 4;
      const r = data[srcOffset] / 255;
      const g = data[srcOffset + 1] / 255;
      const b = data[srcOffset + 2] / 255;
  
      if (format === 'NCHW') {
        rearrangedData[destOffset] = r;
        rearrangedData[destOffset + numPixels] = g;
        rearrangedData[destOffset + 2 * numPixels] = b;
        destOffset++;
      } else if (format === 'NHWC') {
        rearrangedData[destOffset] = r;
        rearrangedData[destOffset + 1] = g;
        rearrangedData[destOffset + 2] = b;
        destOffset += channels;
      } else {
        throw new Error('Invalid format specified.');
      }
    }
  
    const tensorShape = format === 'NCHW' ? [1, channels, height, width] : [1, height, width, channels];
    let tensor = new ort.Tensor("float16", convertToUint16Array(rearrangedData), tensorShape);
  
    return tensor;
  }

async function generate_image() {
    try {
        document.getElementById('status').innerText = "generating ...";

        if (tokenizer === undefined) {
            tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
            tokenizer.pad_token_id = 0;
        }
        let canvases = [];
        await loading;

        for (let j = 0; j < config.images; j++) {
            const div = document.getElementById(`img_div_${j}`);
            div.style.opacity = 0.5
        }

        const { input_ids } = await tokenizer(text.value, { padding: true, max_length: 77, truncation: true, return_tensor: false });

        // text-encoder
        let start = performance.now();
        const { last_hidden_state } = await models.text_encoder.sess.run(
            { "input_ids": new ort.Tensor("int32", input_ids, [1, input_ids.length]) });

        let perf_info = [`text_encoder: ${(performance.now() - start).toFixed(1)}ms`];

        for (let j = 0; j < config.images; j++) {
            const latent_shape = [1, 4, 64, 64];
            let latent = new ort.Tensor(randn_latents(latent_shape, sigma), latent_shape);
            const latent_model_input = scale_model_inputs(latent);

            // unet
            start = performance.now();
            let feed = {
                "sample": new ort.Tensor("float16", convertToUint16Array(latent_model_input.data), latent_model_input.dims),
                "timestep": new ort.Tensor("float16", new Uint16Array([toHalf(999)]), [1]),
                "encoder_hidden_states": last_hidden_state,
            };
            let { out_sample } = await models.unet.sess.run(feed);
            perf_info.push(`unet: ${(performance.now() - start).toFixed(1)}ms`);

            // scheduler
            const new_latents = step(new ort.Tensor("float32", convertToFloat32Array(out_sample.data), out_sample.dims), latent);

            // vae_decoder
            start = performance.now();
            const { sample } = await models.vae_decoder.sess.run({ "latent_sample": new_latents });
            perf_info.push(`vae_decoder: ${(performance.now() - start).toFixed(1)}ms`);
            draw_image(sample, j);

            // safety_checker
            let resized_image_data = resize_image(j, 224, 224);
            let normalized_image_data = normalizeImageData(resized_image_data);
            let safety_checker_feed = {
                "clip_input": get_tensor_from_image(normalized_image_data, "NCHW"),
                "images": get_tensor_from_image(resized_image_data, "NHWC"),
            };
            start = performance.now();
            const { out_images, has_nsfw_concepts } = await models.safety_checker.sess.run(safety_checker_feed);
            perf_info.push(`safety_checker: ${(performance.now() - start).toFixed(1)}ms`);

            log("### has_nsfw_concepts ###: " + has_nsfw_concepts.data[0]);

            let out_image = new ort.Tensor("float32", convertToFloat32Array(out_images.data), out_images.dims);
            draw_out_image(out_image);

            log(perf_info.join(", "))
            perf_info = [];
        }
        // this is a gpu-buffer we own, so we need to dispose it
        last_hidden_state.dispose();
        log("done");
    } catch (e) {
        log(e);
    }
}


async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

document.addEventListener("DOMContentLoaded", () => {
    hasFp16().then((fp16) => {
        if (fp16) {
            loading = load_models(models);
        } else {
            log("Your GPU or Browser doesn't support webgpu/f16");
        }
    });
});

// ref: http://stackoverflow.com/questions/32633585/how-do-you-convert-to-half-floats-in-javascript
const toHalf = (function () {

    var floatView = new Float32Array(1);
    var int32View = new Int32Array(floatView.buffer);

    /* This method is faster than the OpenEXR implementation (very often
     * used, eg. in Ogre), with the additional benefit of rounding, inspired
     * by James Tursa?s half-precision code. */
    return function toHalf(val) {

        floatView[0] = val;
        var x = int32View[0];

        var bits = (x >> 16) & 0x8000; /* Get the sign */
        var m = (x >> 12) & 0x07ff; /* Keep one extra bit for rounding */
        var e = (x >> 23) & 0xff; /* Using int is faster here */

        /* If zero, or denormal, or exponent underflows too much for a denormal
         * half, return signed zero. */
        if (e < 103) {
            return bits;
        }

        /* If NaN, return NaN. If Inf or exponent overflow, return Inf. */
        if (e > 142) {
            bits |= 0x7c00;
            /* If exponent was 0xff and one mantissa bit was set, it means NaN,
             * not Inf, so make sure we set one mantissa bit too. */
            bits |= ((e == 255) ? 0 : 1) && (x & 0x007fffff);
            return bits;
        }

        /* If exponent underflows but not too much, return a denormal */
        if (e < 113) {
            m |= 0x0800;
            /* Extra rounding may overflow and set mantissa to 0 and exponent
             * to 1, which is OK. */
            bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
            return bits;
        }

        bits |= ((e - 112) << 10) | (m >> 1);
        /* Extra rounding. An overflow will set mantissa to 0 and increment
         * the exponent, which is OK. */
        bits += m & 1;
        return bits;
    };

})();

// This function converts a Float16 stored as the bits of a Uint16 into a Javascript Number.
// Adapted from: https://gist.github.com/martinkallman/5049614
// input is a Uint16 (eg, new Uint16Array([value])[0])

export function float16ToNumber(input) {
    // Create a 32 bit DataView to store the input
    const arr = new ArrayBuffer(4);
    const dv = new DataView(arr);

    // Set the Float16 into the last 16 bits of the dataview
    // So our dataView is [00xx]
    dv.setUint16(2, input, false);

    // Get all 32 bits as a 32 bit integer
    // (JS bitwise operations are performed on 32 bit signed integers)
    const asInt32 = dv.getInt32(0, false);

    // All bits aside from the sign
    let rest = asInt32 & 0x7fff;
    // Sign bit
    let sign = asInt32 & 0x8000;
    // Exponent bits
    const exponent = asInt32 & 0x7c00;

    // Shift the non-sign bits into place for a 32 bit Float
    rest <<= 13;
    // Shift the sign bit into place for a 32 bit Float
    sign <<= 16;

    // Adjust bias
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
    rest += 0x38000000;
    // Denormals-as-zero
    rest = (exponent === 0 ? 0 : rest);
    // Re-insert sign bit
    rest |= sign;

    // Set the adjusted float32 (stored as int32) back into the dataview
    dv.setInt32(0, rest, false);

    // Get it back out as a float32 (which js will convert to a Number)
    const asFloat32 = dv.getFloat32(0, false);

    return asFloat32;
}

// convert Uint16Array to Float32Array
export function convertToFloat32Array(fp16_array) {
    const fp32_array = new Float32Array(fp16_array.length);
    for (let i = 0; i < fp32_array.length; i++) {
        fp32_array[i] = float16ToNumber(fp16_array[i]);
    }
    return fp32_array;
}

// convert Float32Array to Uint16Array
export function convertToUint16Array(fp32_array) {
    const fp16_array = new Uint16Array(fp32_array.length);
    for (let i = 0; i < fp16_array.length; i++) {
        fp16_array[i] = toHalf(fp32_array[i]);
    }
    return fp16_array;
}
