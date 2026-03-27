from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

print(ds)                       # shows splits + sizes
print("--------")

print(ds["train"][10]["text"])   # first item
print("-----2nd---")

print(ds["train"][11]["text"])   # second item


print("--------")
# show a few
for i in range(10):
    print(f"{i}: {repr(ds['train'][i]['text'][:120])}")
