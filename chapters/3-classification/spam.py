# %%
import email
import email.policy
import os
import tarfile

import numpy as np
# %%
dir_path = os.path.join("data","spam")
extract = False
if ("easy_ham" not in os.listdir(dir_path)) and ("spam" not in os.listdir(dir_path)):
  extract = True
  print("Extracting...")
# %%
if extract:
  for filename in ["20030228_easy_ham.tar.bz2", "20030228_spam.tar.bz2"]:
    path = os.path.join(dir_path, filename)
    tar_bz2_file = tarfile.open(path)
    tar_bz2_file.extractall(dir_path)
    tar_bz2_file.close()
# %%
ham_dir = os.path.join(dir_path, "easy_ham")
spam_dir = os.path.join(dir_path, "spam")

ham_filenames = [name for name in sorted(os.listdir(ham_dir)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(spam_dir)) if len(name) > 20]
# %%
print("Ham Files:", len(ham_filenames))
print("Spam Files:", len(spam_filenames))
# %%
def load_email(is_spam, filename, dir_path=dir_path):
  """
  Loads an email from filename
  Returns email object
  """
  dir_name = "spam" if is_spam else "easy_ham"
  email_path = os.path.join(dir_path, dir_name, filename)
  with open(email_path, "rb") as f:
    # default email.policy parser
    return email.parser.BytesParser(policy=email.policy.default).parse(f)
# %%
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
# %%
print(ham_emails[1].get_content().strip())
# %%
print(spam_emails[6].get_content().strip())
# %%
def get_email_structure(email):
  if (isinstance(email, str)):
    return email

  payload = email.get_payload()
  if (isinstance(payload, list)):
    multiple = ", ".join([get_email_structure(sub_email) for sub_email in payload])
    return f"multipart({multiple})"
  else:
    return email.get_content_type()
# %%
print(get_email_structure(ham_emails[10]))
# %%
from collections import Counter  # similar to frequency dict

def structures_counter(emails):
  structures = Counter()
  for email in emails:
    structure = get_email_structure(email=email)
    structures[structure] += 1
  return structures
# %%
structures_counter(ham_emails).most_common()
# %%
structures_counter(spam_emails).most_common()
# %%
spam_emails[0].items()
# %%
spam_emails[0]["Subject"]
# %%
from sklearn.model_selection import train_test_split

X = np.array((ham_emails + spam_emails), dtype=object)
y= np.array(([0] * len(ham_emails)) + ([1] * len(spam_emails)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
import re
from html import unescape

def html_to_text(html):
  """
  Parses HTML to plain text
  Return: String of plain text
  """
  text = re.sub("<head.*?>.*?</head>", "", html, flags=re.M | re.S | re.I)
  text = re.sub("<a\s.*?>", " HYPERLINK ", text, flags=re.M | re.S | re.I)
  text = re.sub("<.*?>", "", text, flags=re.M | re.S)
  text = re.sub(r"(\s*\n)+", "\n", text, flags=re.M | re.S)
  return unescape(text)
# %%
html_spam_emails = [email for email in X_train[y_train == 1] if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")
# %%
print(html_to_text(sample_html_spam.get_content().strip())[:1000], "...")
# %%
def email_to_text(email):
  """
  Any email to plain text
  Return: String of plain text
  """
  html = None
  for part in email.walk():
    ctype = part.get_content_type()
    if not ctype in ("text/plain", "text/html"):
      continue
    try:
      content = part.get_content()
    except: # in case of encoding issues
      content = str(part.get_payload())
    if ctype == "text/plain":
      return content
    else:
      html = content
  if html:
    return html_to_text(html)
# %%
print(email_to_text(sample_html_spam)[:100], "...")
# %%
import nltk

stemmer = nltk.PorterStemmer()
for word in ("Computations", "Computing", "Computed", "Computer", "Compute", "Compulsive"):
  print(word, "=>", stemmer.stem(word))
# %%
import urlextract

url_extractor = urlextract.URLExtract()
url_extractor.find_urls("Click here https://github.com/lostvikx and watch https://www.youtube.com/watch?v=bOCHTHkBoAs")
# %%
from sklearn.base import BaseEstimator, TransformerMixin

class EmailToCounter(BaseEstimator, TransformerMixin):
  def __init__(self, lowercase=True, rm_punctuation=True, replace_urls=True, replace_nums=True, stemming=True):
    self.lowercase = lowercase
    self.rm_punctuation = rm_punctuation
    self.replace_urls = replace_urls
    self.replace_nums = replace_nums
    self.stemming = stemming
  
  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    X_trans = []
    for email in X:
      text = email_to_text(email) or ""
      text = text.strip()

      if self.lowercase and (url_extractor is not None):
        text = text.lower()

      if self.replace_urls:
        urls = list(set(url_extractor.find_urls(text)))
        for url in urls:
          text = text.replace(url, " URL ")
        
      if self.replace_nums:
        text = re.sub(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", "NUMBER", text)

      if self.rm_punctuation:
        text = re.sub(r"\W+", " ", text, flags=re.M)

      word_counts = Counter(text.split())

      if self.stemming and (stemmer is not None):
        stemmed_word_counts = Counter()
        for (word, count) in word_counts.items():
          stemmed_word = stemmer.stem(word)
          stemmed_word_counts[stemmed_word] += count
        word_counts = stemmed_word_counts

      X_trans.append(word_counts)

    return np.array(X_trans)
# %%
EmailToCounter().fit_transform(X_train[:3])
# %%
from scipy.sparse import csr_matrix

class CounterToVector(BaseEstimator, TransformerMixin):
  def __init__(self, vocab_size=1000):
    self.vocab_size = vocab_size

  def fit(self, X, y=None):
    total_count = Counter()
    for word_count in X:
      for (word, count) in word_count.items():
        total_count[word] += count # min(count, 10)
    most_common = total_count.most_common()[:self.vocab_size]
    # Zero will be reserved for if not found in the most common ranking.
    self.common_vocab_rank = {word: idx + 1 for idx, (word, count) in enumerate(most_common)}
    return self

  def transform(self, X):
    rows, cols, data = [], [], []
    for (row, word_count) in enumerate(X):
      for (word, count) in word_count.items():
        rows.append(row)
        cols.append(self.common_vocab_rank.get(word, 0))
        data.append(count)
    # print(self.common_vocab_rank)
    return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocab_size + 1))
# %%
vocab_tr = CounterToVector(vocab_size=10)
vocab_tr.fit_transform(EmailToCounter().fit_transform(X_train[:3])).toarray()
# %% [markdown]
# What does the above matrix mean? 2nd email contains 21 words that are not in the top 10 most common words (vocab_size=10). Also, the 2nd email contains 15 words that are the first ranked (most repeated word in the dataset).
# %%
vocab_tr.common_vocab_rank
# %%
from sklearn.pipeline import Pipeline

preprocessor = Pipeline([
  ("email_to_counter", EmailToCounter()),
  ("counter_to_vector", CounterToVector())
])

X_train_tr = preprocessor.fit_transform(X_train)
print(X_train_tr.shape)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver="lbfgs",max_iter=1000,random_state=42)

score = cross_val_score(log_clf,X_train_tr,y_train,cv=5)
# %%
print("Mean Accuracy Score: {:.2f}%".format(score.mean() * 100)) # mean accuracy
# %%
X_test_tr = preprocessor.transform(X_test)

log_clf = LogisticRegression(solver="lbfgs",max_iter=1000,random_state=42)
log_clf.fit(X_train_tr,y_train)

y_pred = log_clf.predict(X_test_tr)
# %%
from sklearn.metrics import precision_score, recall_score

print("Precision: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("Recall: {:.2f}%".format(recall_score(y_test, y_pred) * 100))