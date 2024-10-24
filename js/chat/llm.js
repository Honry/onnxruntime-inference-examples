ort.env.wasm.numThreads = 4;
ort.env.wasm.simd = true;
ort.env.wasm.proxy = false;
ort.env.logLevel = 'warning'; //'error';


function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

function product(shape) {
    let size = 1;
    shape.forEach((element) => {
        size *= element;
    });
    return size;
}

// load file from server or cache
async function fetchAndCache(url) {
    try {
        const cache = await caches.open('onnx');
        let cachedResponse = await cache.match(url);
        if (cachedResponse === undefined) {
            log(`${url} (network)`);
            const buffer = await fetch(url).then(response => response.arrayBuffer());
            try {
                await cache.put(url, new Response(buffer));
            } catch (error) {
                console.error(error);
            }
            return buffer;
        }
        log(`${url} (cached)`);
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        log(`can't fetch ${url}`);
        throw error;
    }
}

// class to handle a large language model on top of onnxruntime-web
export class LLM {
    sess = undefined;
    feed = {};
    prefill_fetches = {};
    decode_fetches = {};
    decode_tmp_fetches = {};
    output_tokens = [];
    eos = 32007;
    need_position_ids = true;
    stop = false;
    kv_dims = [];
    dtype = 'float32';
    max_tokens = 128;
    max_caches = 256;
    attn_mask_len = 384;
    start_len = 0;
    ml_context = undefined;

    constructor(max_tokens, max_caches, dataType) {
        this.dtype = dataType;
        this.max_tokens = max_tokens;
        this.max_caches = max_caches;
        this.attn_mask_len = max_tokens + max_caches;
    }

    async load(model, options, flag = true) {
        const provider = options.provider;
        const verbose = options.verbose;
        const model_path = location.href.includes('github.io') ?
            'https://huggingface.co/lwanming/Phi3-mini-4k-instruct-static/blob/main/'
            : 'models/';

        const type_suffix = this.dtype == 'float16' ? '_fp16' : '';// '_op_simplifiedlayernorm_fp16' : '';
        let model_file_1 = `1_prefill_INT4_128${type_suffix}.onnx`;
        let model_file_2 = `2_decode_INT4_128${type_suffix}.onnx`;
        log(`loading... ${model.name}, ${this.dtype}, ${provider}`);

        const model_bytes_1 = await fetchAndCache(model_path + model_file_1);
        const externaldata_1 = (model.externaldata) ? await fetchAndCache(model_path + model_file_1 + '.data') : false;
        // const model_bytes_1 = model_path + model_file_1;
        // const externaldata_1 = model_path + model_file_1 + '.data';
        // const model_bytes_2 = model_path + model_file_2;
        // const externaldata_2 = model_path + model_file_2 + '.data';
        const model_bytes_2 = await fetchAndCache(model_path + model_file_2);
        const externaldata_2 = (model.externaldata) ? await fetchAndCache(model_path + model_file_2 + '.data') : false; 
        let modelSize_1 = model_bytes_1.byteLength;
        let modelSize_2 = model_bytes_2.byteLength;
        if (externaldata_1) {
            modelSize_1 += externaldata_1.byteLength;
        }
        if (externaldata_2) {
            modelSize_2 += externaldata_2.byteLength;
        }
        log(`model 1 size ${Math.round(modelSize_1 / 1024 / 1024)} MB`);
        log(`model 2 size ${Math.round(modelSize_2 / 1024 / 1024)} MB`);
        this.ml_context = await navigator.ml.createContext({ deviceType: 'gpu' });
        const session_option = {
            executionProviders: [{ name: provider, deviceType: 'gpu', context: this.ml_context }],
            preferredOutputLocation: 'ml-tensor',
            graphOptimizationLevel: 'basic',
        }

        // switch (provider) {
        //     case 'webnn':
        //         // Bind kv cache outputs to ml-tensor
        //         for (let i = 0; i < 32; ++i) {
        //             session_option.preferredOutputLocation[`new_present_key_values.${i}.decoder.key`] = 'ml-tensor';
        //             session_option.preferredOutputLocation[`new_present_key_values.${i}.decoder.value`] = 'ml-tensor';
        //         }
        //         break;
        // }
        if (verbose) {
            session_option.logSeverityLevel = 0;
            session_option.logVerbosityLevel = 0;
        }
        let externalData;
        if (externaldata_1 !== undefined) {
            externalData = [
                {
                    data: externaldata_1,
                    path: model_file_1 + '.data',
                },
            ]
        }
        console.log('create session 1 with option: ', { ...session_option, externalData });
        this.sess_1 = await ort.InferenceSession.create(model_bytes_1, { ...session_option, externalData });

        if (externaldata_2 !== undefined) {
            externalData = [
                {
                    data: externaldata_2,
                    path: model_file_2 + '.data',
                },
            ]
        }
        console.log('create session 2 with option: ', { ...session_option, externalData });
        this.sess_2 = await ort.InferenceSession.create(model_bytes_2, { ...session_option, externalData });
        this.kv_dims = [1, 32, this.max_caches, 96];
        this.num_layers = 32;
        if (!flag) {
            this.initialize_feed();
        }
    }

    async initialize_feed() {
        this.feed = {};
        const input_ml_tensor = await this.ml_context.createTensor({
            dataType: this.dtype,
            shape: this.kv_dims,
            // usage: 
        });
        for (let i = 0; i < this.num_layers; ++i) {
            // The same MLTensor cannot be used more than once as output.
            const prefill_key_ml_tensor = await this.ml_context.createTensor({
                dataType: this.dtype,
                shape: this.kv_dims,
                // usage: 
            });
            const prefill_value_ml_tensor = await this.ml_context.createTensor({
                dataType: this.dtype,
                shape: this.kv_dims,
                // usage: 
            });
            const decode_key_ml_tensor = await this.ml_context.createTensor({
                dataType: this.dtype,
                shape: this.kv_dims,
                // usage: 
            });
            const decode_value_ml_tensor = await this.ml_context.createTensor({
                dataType: this.dtype,
                shape: this.kv_dims,
                // usage: 
            });
            const decode_tmp_key_ml_tensor = await this.ml_context.createTensor({
                dataType: this.dtype,
                shape: this.kv_dims,
                // usage: 
            });
            const decode_tmp_value_ml_tensor = await this.ml_context.createTensor({
                dataType: this.dtype,
                shape: this.kv_dims,
                // usage: 
            });
            // input feed
            this.feed[`past_key_values.${i}.decoder.key`] = ort.Tensor.fromMLTensor(input_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
            this.feed[`past_key_values.${i}.decoder.value`] = ort.Tensor.fromMLTensor(input_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
            // output fetches
            this.prefill_fetches[`new_present_key_values.${i}.decoder.key`] = ort.Tensor.fromMLTensor(prefill_key_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
            this.prefill_fetches[`new_present_key_values.${i}.decoder.value`] = ort.Tensor.fromMLTensor(prefill_value_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
            this.decode_fetches[`new_present_key_values.${i}.decoder.key`] = ort.Tensor.fromMLTensor(decode_key_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
            this.decode_fetches[`new_present_key_values.${i}.decoder.value`] = ort.Tensor.fromMLTensor(decode_value_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
            this.decode_tmp_fetches[`new_present_key_values.${i}.decoder.key`] = ort.Tensor.fromMLTensor(decode_tmp_key_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
            this.decode_tmp_fetches[`new_present_key_values.${i}.decoder.value`] = ort.Tensor.fromMLTensor(decode_tmp_value_ml_tensor,
                { dataType: this.dtype, dims: this.kv_dims });
        }
        this.prefill_tokenid_tensor = await this.ml_context.createTensor({
            dataType: 'int32',
            shape: [1, 1],
            usage: MLTensorUsage.READ,
        });
        this.prefill_fetches['token_id'] = ort.Tensor.fromMLTensor(this.prefill_tokenid_tensor, {
            dataType: 'int32',
            dims: [1, 1],
        });
        this.decode_tokenid_tensor = await this.ml_context.createTensor({
            dataType: 'int32',
            shape: [1, 1],
            usage: MLTensorUsage.READ,
        });
        this.decode_fetches['token_id'] = ort.Tensor.fromMLTensor(this.decode_tokenid_tensor, {
            dataType: 'int32',
            dims: [1,1],
        });
        this.decode_tmp_fetches['token_id'] = ort.Tensor.fromMLTensor(this.decode_tokenid_tensor, {
            dataType: 'int32',
            dims: [1,1],
        });
    }

    // update key value cache
    update_kv_cache(outputs) {
        for (const name in outputs) {
            if (name.includes('new_present_key_values')) {
                let newName = name.replace(name.split('.')[0], 'past_key_values');
                this.feed[newName] = outputs[name];
            }
        }
    }

    // padding input array with 0
    padding_input(input, max_length, reverse = false) {
        if (input.length >= max_length)
            return input;
        const padding_length = max_length - input.length;
        const padding = Array.from({ length: padding_length }, () => 0);
        if (reverse) {
            padding.push(...input);
            return padding;
        } else {
            input.push(...padding);
            return input;
        }
    }
    //
    // tell generate to stop()
    //
    abort() {
        this.stop = true;
    }

    // 
    // prefill prompt and generate tokens, greedy search only
    // tokens => input_ids
    async generate(tokens, continuation, callback) {
        this.output_tokens = [];
        if (!continuation) {
            // clear cache
            this.start_len = 0;
        }
        const token_len = tokens.length;
        let attn_mask;
        if (this.start_len == 0) {
            attn_mask = Array.from({ length: this.max_caches }, () => 0);
        } else {
            attn_mask = Array.from({ length: Math.min(this.start_len, this.max_caches) }, () => 1);
        }

        // padding tokens and attn_mask
        tokens = this.padding_input(tokens, this.max_tokens);
        attn_mask = this.padding_input(attn_mask, this.max_caches, true);
        attn_mask = this.padding_input(attn_mask, this.attn_mask_len);
        for (let i = this.max_caches; i < this.max_caches + token_len; i++) {
            attn_mask[i] = 1;
        }

        const input_ids = new ort.Tensor('int32', Int32Array.from(tokens), [1, this.max_tokens]);
        this.feed['input_ids'] = input_ids;
        const attention_mask = new ort.Tensor('int32', Int32Array.from(attn_mask), [1, this.attn_mask_len]);
        this.feed['attention_mask'] = attention_mask;
        const position_ids = Array.from({ length: this.max_tokens }, (_, i) => this.start_len + i);
        this.feed['position_ids'] = new ort.Tensor('int32', Int32Array.from(position_ids), [1, this.max_tokens]);
        this.stop = false;

        let last_token = 0;

        await this.sess_1.run(this.feed, this.prefill_fetches);
        let token_id = await this.ml_context.readTensor(this.prefill_fetches['token_id'].mlTensorData);
        this.start_len += token_len;
        last_token = (new Int32Array(token_id))[0];
        console.log(last_token);
        this.output_tokens.push(last_token);
        if (callback) {
            callback(this.output_tokens);
        }
        let seqlen = token_len;
        this.update_kv_cache(this.prefill_fetches);

        while (last_token != this.eos && !this.stop) {
            this.feed['input_ids'] = new ort.Tensor('int32', Int32Array.from([last_token]), [1, 1]);
            attn_mask = Array.from({ length: Math.min(this.start_len, this.max_caches) }, () => 1);
            attn_mask = this.padding_input(attn_mask, this.max_caches, true);
            attn_mask = this.padding_input(attn_mask, this.max_caches + 1);
            attn_mask[this.max_caches] = 1;
            this.feed['attention_mask'] = new ort.Tensor('int32', new Int32Array(attn_mask), [1, this.max_caches + 1]);
            this.feed['position_ids'] = new ort.Tensor('int32', Int32Array.from([this.start_len]), [1, 1]);

            const fetches = this.start_len % 2 == 0 ? this.decode_fetches : this.decode_tmp_fetches;
            await this.sess_2.run(this.feed, fetches);
            console.time('read decode token');
            let token_id = await this.ml_context.readTensor(fetches['token_id'].mlTensorData);
            console.timeEnd('read decode token');

            last_token = (new Int32Array(token_id))[0];
            console.log('next token: ', last_token);
            this.output_tokens.push(last_token);
            if (callback) {
                callback(this.output_tokens);
            }
            this.update_kv_cache(fetches);
            this.start_len += 1;
            seqlen += 1;
        }
        return this.output_tokens;
    }
}