import nltk
import pandas as pd
import datasets
import tensorflow as tf
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
sys.path.append('/home/ubuntu/Review-Summerization/helper_function.py')
from helper_function import set_seeds, print_gpu_utilization

# PATH = '/home/ubuntu/Review-Summerization/clean_data/test.csv'
PATH = "/home/ubuntu/Review-Summerization/clean_data/item_reviews.csv"
SEED=666
TEST_RATIO = 0.2
BATCH_SIZE = 16
# BATCH_SIZE = 64
encoder_max_length = 512
# decoder_max_length = 100
# encoder_max_length = 800
decoder_max_length = 64
ckpt = 'bert-base-cased'
# ckpt='t5-small'
# model_dir = "t5-small"
# model_dir = "/home/ubuntu/Review-Summerization/model/T5"
model_dir = "/home/ubuntu/Review-Summerization/checkpoint/fullbert/checkpoint-10800"
# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.to("cuda")
torch.cuda.empty_cache()

print_gpu_utilization()
metric = datasets.load_metric("rouge")


def compute_metrics(pred_str, label_str):
    """
    Return ROUGE metric
    :param pred_str: model generated str
    :param label_str: target str
    :return:
    """
    # Rouge expects a newline after each sentence
    pred_str = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in pred_str]
    label_str = ["\n".join(nltk.sent_tokenize(label.strip())) for label in label_str]

    result = {}
    result['rouge1'] = \
    metric.compute(predictions=pred_str, references=label_str, use_stemmer=True, rouge_types=["rouge1"])["rouge1"].mid
    result['rouge2'] = \
    metric.compute(predictions=pred_str, references=label_str, use_stemmer=True, rouge_types=["rouge2"])["rouge2"].mid
    result['rougeL'] = \
    metric.compute(predictions=pred_str, references=label_str, use_stemmer=True, rouge_types=["rougeL"])["rougeL"].mid

    return result



def generate_summary(batch):
    """
    return model generated output
    :param batch: batched input
    :return:
    """
    if ckpt in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""
    inputs = [prefix + doc for doc in batch['text']]
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = tokenizer(inputs, padding=True, truncation=True, max_length=encoder_max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    with torch.no_grad():
        # top-k & top-p sample
        tf.random.set_seed(0)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            # length_penalty=2,
            max_length=decoder_max_length,
            # min_length= 10,
            # num_beams=3,
            no_repeat_ngram_size=2,
            # no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.92,
            # num_return_sequences=3
        )
        # print(outputs)
    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str
    # print(type(batch), type(output_str), batch)
    return batch


df = pd.read_csv(PATH, header=0, index_col=[0])
test_data = Dataset.from_pandas(df, preserve_index=False)

results = test_data.map(generate_summary, batched=True, batch_size=BATCH_SIZE)
pred_str = results["pred"]
label_str = results["summary"]


# remove stopwords before evaluation
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
#
def remove_stop_words(corpus):
    result = []
    for summary in corpus:
        corp = summary.split(' ')
        rm_stop = [w for w in corp if w not in stop_words]
        rm_stop = " ".join(rm_stop).strip()
        if rm_stop != "":
            result.append(rm_stop)
        else:
            result.append(summary)
    return result
#
# pred_str = remove_stop_words(pred_str)
# label_str = remove_stop_words(label_str)

print("result metric:")
score = compute_metrics(pred_str, label_str)
print("ROUGE 1 SCORE: ",score['rouge1'])
print("ROUGE 2 SCORE: ",score['rouge2'])
print("ROUGE L SCORE: ",score['rougeL'])
output = pd.DataFrame({'reviewText': results['text'], 'predSummary':results['pred'], 'actualSummary':results['summary']})
# output.to_csv('output/T5_summary.csv')

# bert
# 0,0.036822612976016925,0.009259701389587018,0.03432397232615782
# 1,0.30850046685340804,0.10975000000000001,0.2952617848193296
# 2,0.06278757063952087,0.01643812441298275,0.05865568417074169
#
# ROUGE 1 SCORE:  Score(precision=0.08713425441261022, recall=0.21237246787101643, fmeasure=0.10427800024485728)
# ROUGE 2 SCORE:  Score(precision=0.049848501773376734, recall=0.09715565594106625, fmeasure=0.05576334830472214)
# ROUGE L SCORE:  Score(precision=0.08366079422697068, recall=0.20089801137103697, fmeasure=0.09940487722195637)

# T5 - baeline
# ROUGE 1 SCORE:  Score(precision=0.08976907136045903, recall=0.23369877669905004, fmeasure=0.11518994583043468)
# ROUGE 2 SCORE:  Score(precision=0.044849471683858166, recall=0.09658222854232953, fmeasure=0.05364179689506758)
# ROUGE L SCORE:  Score(precision=0.08383736194387868, recall=0.2178398857680645, fmeasure=0.10726622097617411)

# T5 - freeze decoder regular - T5_summary
# ROUGE 1 SCORE:  Score(precision=0.1042398688213995, recall=0.22586533407579118, fmeasure=0.12638363675569567)
# ROUGE 2 SCORE:  Score(precision=0.05815841286225652, recall=0.10420943224685245, fmeasure=0.06555518410414296)
# ROUGE L SCORE:  Score(precision=0.09950363248621932, recall=0.2140075017095628, fmeasure=0.12015505954367148)

# T5 - freeze decoder topk(models) - T5_Topk_summary
# ROUGE 1 SCORE:  Score(precision=0.11541883890219187, recall=0.155093420910352, fmeasure=0.11416971537437634)
# ROUGE 2 SCORE:  Score(precision=0.05626714803061421, recall=0.06512508862320987, fmeasure=0.05184892404589099)
# ROUGE L SCORE:  Score(precision=0.11108487778603378, recall=0.148836434276937, fmeasure=0.10962168687762905)

# T5 - top layer unfreeze(T5) - T5_summary1
# ROUGE 1 SCORE:  Score(precision=0.11597057063289304, recall=0.15053375031656935, fmeasure=0.11272986839212631)
# ROUGE 2 SCORE:  Score(precision=0.054871493078342104, recall=0.06015626102669518, fmeasure=0.04951596096342424)
# ROUGE L SCORE:  Score(precision=0.11157713916889146, recall=0.14398599442338356, fmeasure=0.10809128436309318)

# T5 - top two layer unfreeze(T5-2)
# ROUGE 1 SCORE:  Score(precision=0.12804244365111173, recall=0.15745614708213754, fmeasure=0.12110307177929257)
# ROUGE 2 SCORE:  Score(precision=0.06116444359975244, recall=0.0686125585063802, fmeasure=0.05581745544145077)
# ROUGE L SCORE:  Score(precision=0.12389385492981209, recall=0.15148504221811265, fmeasure=0.11685261158916396)

# T5 - freeze encoder(T5Decoder)
# ROUGE 1 SCORE:  Score(precision=0.08106375205009067, recall=0.26090331462033256, fmeasure=0.1116409906490468)
# ROUGE 2 SCORE:  Score(precision=0.043444771642893336, recall=0.11587400905354081, fmeasure=0.05642097234014582)
# ROUGE L SCORE:  Score(precision=0.07602575945011218, recall=0.24368034418909862, fmeasure=0.10432372136761692)

# T5 - top 2 layer unfreeze(T5Top2)
# item summary
# ROUGE 1 SCORE:  Score(precision=0.31348812097927176, recall=0.2599672906709703, fmeasure=0.2543790486588796)
# ROUGE 2 SCORE:  Score(precision=0.2045102368105386, recall=0.1711325167054203, fmeasure=0.17133615933482355)
# ROUGE L SCORE:  Score(precision=0.3045722088424222, recall=0.25512361605701606, fmeasure=0.24955894674962392)
#
# ROUGE 1 SCORE:  Score(precision=0.24163003478680062, recall=0.4160623321964044, fmeasure=0.27016058171884727)
# ROUGE 2 SCORE:  Score(precision=0.18713390712126987, recall=0.26694683577005573, fmeasure=0.2007601129383211)
# ROUGE L SCORE:  Score(precision=0.23720288833639314, recall=0.4013594323118639, fmeasure=0.2639807425288373)
# with early stopping and no_repeat_ngram_size=1, beam=3
# ROUGE 1 SCORE:  Score(precision=0.3301402698393804, recall=0.32220705202047406, fmeasure=0.3107072937211153)
# ROUGE 2 SCORE:  Score(precision=0.24225655564029586, recall=0.2339223195864663, fmeasure=0.23091159928662924)
# ROUGE L SCORE:  Score(precision=0.32598711354069915, recall=0.3190266882571502, fmeasure=0.3076856015510181)
# without early stopping
# ROUGE 1 SCORE:  Score(precision=0.26651137136082337, recall=0.3750884394084244, fmeasure=0.28658859262846603)
# ROUGE 2 SCORE:  Score(precision=0.20117236499611368, recall=0.24605795536571273, fmeasure=0.20854257660513792)
# ROUGE L SCORE:  Score(precision=0.260561233162693, recall=0.36302483284740283, fmeasure=0.27929750904064954)
# topk=50 - T5_sampling_summary
#ROUGE 1 SCORE:  Score(precision=0.2863993076341744, recall=0.29010964741466205, fmeasure=0.2717416862655129)
# ROUGE 2 SCORE:  Score(precision=0.196604662522247, recall=0.19780301048660046, fmeasure=0.19070395256348283)
# ROUGE L SCORE:  Score(precision=0.2829191835619125, recall=0.28623133024795844, fmeasure=0.26863664412066623)

# input size = 1024 nbeam=3 model T5 - T5_summary1
# ROUGE 1 SCORE:  Score(precision=0.3195540173678044, recall=0.32394619757707255, fmeasure=0.305452883031906)
# ROUGE 2 SCORE:  Score(precision=0.2355076570290019, recall=0.23206756657404043, fmeasure=0.22610689621327013)
# ROUGE L SCORE:  Score(precision=0.3161469080724135, recall=0.3201918445544364, fmeasure=0.30238844103098783)

# bert
# ROUGE 1 SCORE:  Score(precision=0.039780353849768156, recall=0.48011971375214724, fmeasure=0.07051167955813213)
# ROUGE 2 SCORE:  Score(precision=0.012457867192885715, recall=0.2821559005493792, fmeasure=0.022822019680081865)
# ROUGE L SCORE:  Score(precision=0.03625462919786011, recall=0.4587101792404304, fmeasure=0.06467920894987747)
#
# ROUGE 1 SCORE:  Score(precision=0.05150657079844559, recall=0.3899701284343593, fmeasure=0.07991874044873495)
# ROUGE 2 SCORE:  Score(precision=0.012220511733918804, recall=0.18176944527381672, fmeasure=0.0202773300689857)
# ROUGE L SCORE:  Score(precision=0.042382613882508115, recall=0.35991353082114463, fmeasure=0.06800235522170081)

# ROUGE 1 SCORE:  Score(precision=0.05249274028768869, recall=0.3953357015534693, fmeasure=0.08179036658243183)
# ROUGE 2 SCORE:  Score(precision=0.012904994097877542, recall=0.1885819460893132, fmeasure=0.021443529601407288)
# ROUGE L SCORE:  Score(precision=0.04396999096216833, recall=0.366493477539247, fmeasure=0.07045783563324023)