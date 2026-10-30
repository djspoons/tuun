
open std;
use synth.sub;

#{level_db=-60}
rand = fn() => noise * fixed([1]) & 0 | filter([1], [-1]);

#{level_db=-60}
maybe = fn(w, threshold=0.5) => alt(rand() / 2 + threshold - 0.5, w, 0);

#{level_db=0}
b = sub.kick_drum()(Q) | unseq();

#{loop_after_beats=4, level_db=0}
_ = <map(fn (w) => w * Qw, [b, maybe(b, threshold=0.2), maybe(b), maybe(b, threshold=0.7)])>;

#{level_db=0}
_ = alt(rand(), $220, $330) * Qw;
