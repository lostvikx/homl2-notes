# %%
import email
import email.policy
import os
import tarfile

import numpy as np
# %%
dir_path = os.path.join("data","spam")
for filename in ["easy_ham.tar.bz2", "spam.tar.bz2"]:
  path = os.path.join(dir_path, filename)
  tar_bz2_file = tarfile.open(path)
  tar_bz2_file.extractall(dir_path)
  tar_bz2_file.close()
# %%
ham_dir = os.path.join(dir_path, "easy_ham")
spam_dir = os.path.join(dir_path, "spam")

ham_filenames = [name for name in sorted(os.listdir(ham_dir))]
spam_filenames = [name for name in sorted(os.listdir(spam_dir))]
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
    return email.parser.BytesParser(policy=email.policy.default).parse(f)
# %%
ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]
# %%
print(ham_emails[5].get_content().strip())
# %%
print(spam_emails[5].get_content().strip())
# %%
def get_email_structure(email):
  if (isinstance(email,str)):
    return email
  elif (isinstance(email,list)):
    payload = email.get_payload()
    return f"multipart({','.join([get_email_structure(sub_email) for sub_email in payload])})"
  else:
    return email.get_content_type()
# %%
print(get_email_structure(ham_emails[5]))
# %%
