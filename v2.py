from transformers import BartTokenizer, BartForConditionalGeneration

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

def summarize_legal_text(text, max_input_length=1024, max_output_length=150):
    inputs = tokenizer.batch_encode_plus([text], return_tensors="pt", max_length=max_input_length, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=max_output_length, min_length=20, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
