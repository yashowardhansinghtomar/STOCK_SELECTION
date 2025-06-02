from core.skiplist.skiplist import get_skiplist

# print("🧾 Current Skiplist:")
# for symbol in get_skiplist():
#     print(symbol)


skiplist = get_skiplist()

for symbol in skiplist:
    print("•", symbol)
print(f"\n🧾 Current Skiplist ({len(skiplist)} stocks):")
