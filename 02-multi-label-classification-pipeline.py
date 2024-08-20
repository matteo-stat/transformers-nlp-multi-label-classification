from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import json

# ----------------------------------------------------------------------------------------
# hugging face model name
model_name = 'FacebookAI/xlm-roberta-base'

# path where fine tuned model was saved
model_name_fine_tuned = 'models/multi-label-classification/FacebookAI/xlm-roberta-base-fine-tuned-en-it-6epochs-32batch-model'

# use cpu (set false for nvidia gpu)
use_cpu = False
# ----------------------------------------------------------------------------------------

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# load labels mapping
with open('data/multi-label-classification/labels-mapping-en-it.json', 'r') as file:
    label2id = json.load(file)
id2label = {id: label for label, id in label2id.items()}

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_fine_tuned,
    num_labels=len(id2label),
    problem_type='multi_label_classification',
    id2label=id2label,
    label2id=label2id
)

# set device
if use_cpu == False:
    model.to('cuda')

# load pipeline
text_classifier = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    device='cuda' if use_cpu == False else 'cpu',
    top_k=None
)

# use pipeline
single_prediction = text_classifier('put some text here')

multiple_predictions = text_classifier(['put first text here', 'put second text here', '...'])
