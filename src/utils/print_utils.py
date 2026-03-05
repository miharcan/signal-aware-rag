def print_section(title):
    print("\n" + "="*30)
    print(title)
    print("="*30)


def print_subsection(title):
    print("\n" + title)
    print("-"*len(title))


def print_docs(docs):
    for d in docs:
        print(f"{d['company']:<15} {d['growth']:>7.3f}")