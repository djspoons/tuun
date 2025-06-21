
x = 220
base = [x, x * 3, x * 5, x * 7, x * 9]
print("Base frequencies:", base)
x = 220 * 5/4
pure = [x, x * 3, x * 5, x * 7, x * 9]
x = 220 * pow(2, 4/12)
equal = [x, x * 3, x * 5, x * 7, x * 9]

def compute_diffs(base, other, diffs):
    for (i, f) in enumerate(other):
        diff = 22000
        diffs.append([diff, f, 0])
        for (j, g) in enumerate(base):
            if abs(f - g) < diff:
                diff = abs(f - g)
                diffs[i] = [diff, f, g]

pure_diffs = []
compute_diffs(base, pure, pure_diffs)
equal_diffs = []
compute_diffs(base, equal, equal_diffs)

def ERB(f):
    return 24.7 * (4.37 * f / 1000 + 1)

def print_diffs(diffs):
    for x in diffs:
        diff = x[0]
        f = x[1]
        g = x[2]
        erb = ERB((f+g)/2)
        print(f"{diff:.2f} Hz, ({f:.2f} Hz, {g:.2f} Hz), ERB: {erb:.2f} Hz, fraction of ERB: {diff/erb:.2f}")

print("pure diffs:")
print_diffs(pure_diffs)

print("equal diffs:")
print_diffs(equal_diffs)

print(220)
print(ERB(220)*1 + 220)