from flask import Flask,render_template,jsonify,request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras


# ================================ LOAD ================================

# Load model
transformer = keras.models.load_model(
    "transformer_model.keras",
    custom_objects={
        "Transformer": Transformer,
        "Encoder": Encoder,
        "Decoder": Decoder,
        "EncoderLayer": EncoderLayer,
        "DecoderLayer": DecoderLayer,
        "MultiHeadAttention": MultiHeadAttention,
        "CustomSchedule": CustomSchedule
    }
)

# Load tokenizers
with open("tokenizers.pkl", "rb") as f:
    tokenizers = pickle.load(f)

document_tokenizer = tokenizers["document_tokenizer"]
summary_tokenizer = tokenizers["summary_tokenizer"]

# Load config
with open("config.pkl", "rb") as f:
    config = pickle.load(f)

encoder_maxlen = config["encoder_maxlen"]
decoder_maxlen = config["decoder_maxlen"]


# ============================ USE (INFERENCE) ===========================

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def summarize(text):
    inp = document_tokenizer.texts_to_sequences([text])
    inp = tf.keras.preprocessing.sequence.pad_sequences(
        inp, maxlen=encoder_maxlen, padding="post", truncating="post"
    )

    encoder_input = tf.expand_dims(inp[0], 0)

    decoder_input = [summary_tokenizer.word_index["<go>"]]
    output = tf.expand_dims(decoder_input, 0)

    for _ in range(decoder_maxlen):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output
        )

        predictions, _ = transformer(
            encoder_input,
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )

        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)

        if predicted_id == summary_tokenizer.word_index["<stop>"]:
            break

        output = tf.concat([output, predicted_id], axis=-1)

    result = output.numpy()[0][1:]  # remove <go>
    return summary_tokenizer.sequences_to_texts([result])[0]


# ================================= ROUTING ==================================

@app.route('/')
@app.route("/home")
def home_page():
    return render_template('home.html')

@app.route("/about")
def about_page():
    return render_template('about_us.html')

@app.route("/signin")
def sign_in():
    return render_template('signin.html')

@app.route("/signup")
def sign_up():
    return render_template('signup.html')

@app.route("/predict", methods=['POST'])
def predict():
    # Accept text input from the form
    input_text = request.form.get('inputtext') or \
                 request.form.get('input_text') or \
                 request.form.get('text') or ''
    
    if not input_text:
        return render_template('output.html', predicted_text="No input provided.")

    try:
        paraphrase_text = summarize(input_text)
    except Exception as e:
        paraphrase_text = f"Error generating summary: {e}"

    return render_template('output.html', predicted_text=paraphrase_text)
