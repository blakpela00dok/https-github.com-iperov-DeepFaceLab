import ymauto.MergeDefault as MD

md = MD.MergeArgs("config.json")

print(md.g("sandy", "name"))
print(md.g("beauty", "deep", "gender"))
# print(md.g("version1"))
# print(md.g("deep", "method"))
# print(md.g("deep", "methoda", "PP"))
# print([v + '真神' for v in md.g("array")])
