from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer, AutoOptimizationConfig

# ----------------------------------------------------------------------------------------
# path where you saved your fine tuned model
model_path = 'models/multi-label-classification/FacebookAI/xlm-roberta-base-fine-tuned-en-it-6epochs-32batch-model'
# ----------------------------------------------------------------------------------------

# load model
model = ORTModelForSequenceClassification.from_pretrained(
    model_id=model_path,
    export=True
)

# create the model optimizer
optimizer = ORTOptimizer.from_pretrained(model)

# create appropriate configuration for the choosen optimization strategy
optimization_config = AutoOptimizationConfig.O2()

# optimize and save the model
optimizer.optimize(
    save_dir=f'{model_path}-inference-optimized',
    optimization_config=optimization_config
)
