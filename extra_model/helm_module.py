import torch
from transformers import AutoTokenizer, BartForSequenceClassification
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("D:/Project/MuCoCP2.0/Stage_5_Lariat/HELM-Binary",
                                          ignore_mismatched_sizes=True)
bart = BartForSequenceClassification.from_pretrained("D:/Project/MuCoCP2.0/Stage_5_Lariat/HELM-Binary",
                                                     ignore_mismatched_sizes=True)
bart.resize_token_embeddings(len(tokenizer))
bart.load_state_dict(torch.load(
    'D:/Project/MuCoCP2.0/Stage_5_Lariat/model/Score_Function/Binary/Binary-Bart-HELM-fold1.ckpt'))

helm_encoder = bart.model.encoder



















