# Research: the Launchkey "repeat" button (ticket 01)

Answers [`tickets/01-launchkey-repeat-button.md`](../tickets/01-launchkey-repeat-button.md).
Researched 2026-10-26 against tuun at commit `6e1b64e`.

## How far to trust this note

Novation's own documents could not be read directly from this session. The egress
proxy refused every Novation/Focusrite host (`fael-downloads-prod.focusrite.com`,
`userguides.novationmusic.com`, `downloads.novationmusic.com`, `novationmusic.com`),
and also the mirrors (`scribd.com`, `manuals.plus`). So each claim below is tagged
with the best source it could be checked against:

- **[tuun]**: what tuun's code already does on the hardware (`src/lib/launchkey.rs`,
  `src/lib/midi_input.rs`). These are the strongest checks available, because the code
  runs against a real device.
- **[Ardour]**: Ardour's open-source Launchkey MK4 surface,
  [`libs/surfaces/launchkey_4/launchkey_4.h`](https://github.com/Ardour/ardour/blob/master/libs/surfaces/launchkey_4/launchkey_4.h)
  and [`launchkey_4.cc`](https://github.com/Ardour/ardour/blob/master/libs/surfaces/launchkey_4/launchkey_4.cc)
  (read from raw.githubusercontent.com, `master`).
- **[StudioPro]**: [miraer/launchkey-mk4-studio-pro](https://github.com/miraer/launchkey-mk4-studio-pro),
  file `Novation Launchkey MK4/LaunchkeyMK4Protocol.js`. Its header says it follows the
  *Launchkey MK4 Programmer's Reference Guide v3.0*, "checked on hardware" (a 49), and
  cross-checked against the Bitwig and Zynthian MK4 code. It is third-party but very
  detailed.
- **[Novation, via search]**: Novation pages known only from search-engine excerpts:
  - the [Programmer's Reference Guide v2.0 PDF](https://fael-downloads-prod.focusrite.com/customer/prod/downloads/launchkey_mk4_programmer_s_reference_guide_v2_en.pdf)
  - [DAW control with the Launchkey 37](https://userguides.novationmusic.com/hc/en-gb/articles/27612230389138-DAW-control-with-the-Launchkey-37)
  - the [Launchkey 37 user guide v3.0](https://fael-downloads-prod.focusrite.com/customer/prod/downloads/launchkey_mk4_37_user_guide_v3_en.pdf)
  - the [Launchkey 37 MK4 product page](https://us.novationmusic.com/products/launchkey-37)

  Excerpts are second-hand, so whoever has the PDF should check these points against it
  (see "To verify" at the end).

## Which model tuun supports

Only the **Launchkey MK4 37** (the full-size MK4, not the Mini). **[tuun]** It opens the
ports by exact name: `Launchkey MK4 37 DAW In` (where it sends), `Launchkey MK4 37 DAW Out`
(DAW-protocol input) and `Launchkey MK4 37 MIDI Out` (keys) (`launchkey.rs`, in
`Launchkey::new`). It enters DAW mode with `9F 0C 7F` and leaves it with `9F 0C 00`.
The source comments cite pages of the "MK4 reference".

## 1. Which physical control is the "repeat" button

**The Launchkey MK4 37 has no button labelled "Repeat" or "Note Repeat". The control
that fits is the transport button labelled Loop.** This is an inference, so it is open
to question:

- **[Novation, via search]** The product page lists "play/stop/record/loop" transport
  buttons for the 37. The "DAW control with the Launchkey 37" page says Loop toggles the
  DAW's loop, and its Shift column for Loop is blank.
- **[Ardour] [StudioPro]** Both list one loop control, `Loop = 0x76`, and no repeat
  control. Neither has a CC named repeat. `grep -i repeat` finds nothing in either source.
- An [Attack Magazine review of the 49 MK4](https://www.attackmagazine.com/reviews/gear-software/novation-launchkey-49-mk4/)
  (secondary) says the MK4 has no MPC-style Note Repeat. The MK4 does have an **Arp**
  button. Arp is a firmware feature on the keys, not a repeat button.

Points that are still unclear:

- The tuun map uses "Repeat" for the domain concept, and `agents/CONTEXT.md` lists
  *loop* as a word to avoid. The hardware button is still labelled **Loop**, so code
  and docs should call the physical control "the Loop button" and use it as the Repeat
  prefix.
- The 37's panel labels include **Function**, the button right of the pads. tuun
  already uses it to cycle the app-wide repeat (`Event::PadFunctionDown` →
  `Action::CycleRepeatAfterMeasures`), and the map frees it. Someone may have meant this
  button by "the repeat button", because it is tuun's repeat button today. Function
  sends CC 105 (`0x69`). Its LED is RGB from the palette, set by sending CC 105 back on
  channel 1 (`set_pad_function_color`). **[tuun]** Both buttons would work. Loop is the
  only one whose label means repeating.
- [StudioPro]'s README says "The 25 and 37 have no faders and only Play and Record as
  transport buttons". This contradicts both Novation's product page and tuun's own
  `StopDown` on CC 116, which is wired up and has tests. *Hypothesis (unverified):* that
  README line is wrong for the 37, which does have physical Stop and Loop buttons.
  Before building on Loop, check the panel of the real unit.

## 2. What Loop sends in DAW mode

| Property | Value | Source |
|---|---|---|
| Message | Control Change, controller **118 (`0x76`)** | [Ardour] `Loop = 0x76`; [StudioPro] `kButtonLoop = 0x76` |
| Channel | **1** (status `B0`) | [StudioPro] "Buttons (CC, channel 1)"; [Ardour] handles it in the same CC switch as Play/Stop/Record |
| Port | The DAW port: **`Launchkey MK4 37 DAW Out`**, where tuun's `DAWState` listens | [StudioPro] DAW ports; [tuun] port names |
| Press | value **127** | [Ardour] acts on `ev->value == 0x7f` |
| Release | value **0** | *Hypothesis (unverified) for Loop specifically.* Its neighbours on the transport row (Play 115, Stop 116, Record 117) send 127/0 on press/release, and tuun matches both values for each [tuun]. [Ardour] treats any value except 127 as a release. |

The neighbouring CCs line up with tuun's code: Play `0x73`=115, Stop `0x74`=116,
Record `0x75`=117 are exactly tuun's `(115, …)`, `(116, …)`, `RECORD_CC = 117`.
**[tuun]** So Loop being the next CC, 118, fits the pattern, and two independent
implementations give the same number.

**What tuun does with it today: nothing.** `DAWState::decode` has no arm for 118, so a
Loop press falls through to `"Ignoring message Controller(118, 127) on channel 0"`.
**[tuun]** That log line gives a cheap hardware check: run tuun, press Loop, read the
log.

The check is not channel-filtered: tuun's button `match` ignores the channel, apart
from the encoder, Shift and mode CCs. **[tuun]**

### Shift and Loop

- tuun follows Shift from channel 7, CC 63 (`0x3F`), press and release (`SHIFT_CC`,
  `shift_held`). **[tuun]** This agrees with [StudioPro]'s `kFeatureShift = 0x3F`. (A
  second report CC `0x48` also exists there. [Ardour]'s `Shift = 0x13` disagrees and is
  probably wrong.)
- With Shift held, some buttons send a different CC: Track ◀/▶ go from `0x67`/`0x66`
  to `0x6D`/`0x6C`. **[StudioPro]** tuun already relies on this: 108/109 →
  `NextTrackBankDown`/`PreviousTrackBankDown`. **[tuun]** *Hypothesis (unverified):*
  Shift+Loop still sends CC 118, because Novation lists no Shift function for Loop. If
  tuun needs Shift+Loop, it can use its `shift_held` flag, as Undo/Redo does.

## 3. Can the host set the Loop LED?

Yes, but it is very likely a **single-colour (white) LED**, so the host can set its
brightness and not its colour.

- **[Novation, via search]** The Programmer's Reference says that monochrome LEDs take
  "a CC on channel 4, the CC number is the LED index, the value is the brightness"
  (0–127). The guide's own example is `93h 73h 7Fh`, with index `0x73` = Play. This
  shows that the transport buttons are monochrome. (The example uses a note-on status,
  `93`, while the text says CC. That is an inconsistency in the guide itself.)
- **[tuun]** tuun already uses this mechanism for Record: `set_brightness(RECORD_CC, …)`
  sends `B3 75 vv` (`BRIGHTNESS_CHANNEL = 3`, 0-indexed). Metronome works the same way,
  and is marked *Hypothesis (unverified on hardware)* in the code.
- **[StudioPro]** drives the Loop, Play, Stop, Record and Metronome LEDs differently:
  `B0 76 <palette index>`, using only the white entries 3 (on) and 1 (dim). [Ardour]
  uses `B0`/`B1`/`B2` (static/flash/pulse colour) for buttons in general. *Hypothesis
  (unverified):* the firmware maps a palette value on channel 1 to some white level for
  monochrome LEDs. Both routes might work, but channel 4 brightness is the route
  Novation documents for monochrome LEDs, and tuun already uses it.
- *Hypothesis (unverified):* channels 2 and 3 (flash/pulse) also apply to Loop. If so,
  a pulsing Loop LED could show "prefix pending". If not, a switch between two
  brightness levels can stand in for the pulse.

What to do: add `const LOOP_CC: u8 = 118;` and `set_loop_brightness`, built like
`set_record_brightness`. Label it as a hypothesis until it has been seen on the
hardware.

## 4. Does the firmware use Loop itself?

- **In DAW mode, no conflict is documented.** Novation's 37 DAW-control page lists
  Loop only as a DAW transport function (loop on/off), and lists no Shift function for
  it. **[Novation, via search]** In DAW mode the button only sends its CC to the host,
  and the host decides what it means. This matches how tuun uses Play, Stop and Record.
- **No firmware Note Repeat** on the MK4, so pads do not repeat when the button is held.
  (See section 1. The source is a secondary review, so this is a hypothesis until it
  has been checked against the user guide.)
- **Real firmware conflicts are nearby, not on Loop:**
  - **Shift + pads** select pad and encoder modes on the device. **[StudioPro]** says
    "the mode is selected on the Launchkey (Shift + pads)", and tuun's `PadModeChanged`
    handling exists because of this. So a Repeat prefix must not be a Shift chord on the
    pads: the firmware takes those presses and they never reach tuun as
    `DAWTopPadDown`/`DAWBottomPadDown`.
  - **Arp, Scale, Chord Map, Fixed Chord** are firmware features on the keys.
    **[Novation, via search]** They are not on the pads, so they do not interfere.
  - *Hypothesis (unverified):* outside DAW mode (standalone), transport buttons send CCs
    on channel 16 on the MIDI port. This does not matter for tuun, which keeps the
    device in DAW mode while it runs.

## 5. Pad layout for showing a value

All **[tuun]**, and agrees with [StudioPro]:

- **16 pads, two rows of 8.** In the DAW pad layout they send **note-on, channel 1,
  DAW port**:
  - top row: notes **96–103** (`0x60–0x67`, `DAW_PAD_TOP_ROW_OFFSET`)
  - bottom row: notes **112–119** (`0x70–0x77`, `DAW_PAD_BOTTOM_ROW_OFFSET`)

  Index 0 is the leftmost pad.
- **Per-pad colour:** full RGB (each 0–127) through SysEx
  `F0 00 20 29 02 14 01 43 <pad note> <r> <g> <b> F7`, using
  `set_daw_top_pad_color` / `set_daw_bottom_pad_color`. [StudioPro] also documents
  palette colours by note-on on channel 1 (static), 2 (flash) and 3 (pulse). tuun does
  not use these yet, and they would give a blinking "current value" pad for free.
- **Input:** tuun turns only note-ons with velocity > 0 into `DAWTopPadDown` /
  `DAWBottomPadDown`. Releases and polyphonic aftertouch are dropped. A pad-based
  editor therefore sees a press only, which is enough to pick a value.
- **Ownership:** tuun owns the pads only while `pad_mode == PadMode::DAW` (reported on
  channel 7, CC 29 = 2). `update_launchkey_state` stops before drawing pads in any other
  pad mode. A Repeat editor has to follow the same rule, or say what happens when the
  user leaves DAW pad mode while the prefix is pending.
- **What fits:** 16 cells give, for example, 1–16 beats one-to-one (top row 1–8, bottom
  row 9–16), with "no repeat" placed somewhere else (for example on the Loop button
  itself, or on Function). Values a pad cannot show can go on the device screen. tuun
  can already write text there (`set_daw_mode_display`,
  `set_encoder_display`). This layout choice is for ticket 05.
- The up/down buttons left of the pads (CC 106/107, `PadPageUpDown`/`PadPageDownDown`)
  and `>` (CC 104, `LaunchDown`) are also free to page through or confirm values
  while the prefix is pending. **[tuun]**

## To verify (cheap checks for whoever has the hardware or the PDF)

1. Press Loop with tuun running. Expect `On channel 0, got controller 118 with value 127`,
   then `… value 0` on release. This settles the CC, the channel and the press/release
   behaviour.
2. Send `B3 76 7F` and `B3 76 00` on DAW In, and see whether the Loop LED goes bright
   and dim. Also try `B0 76 03` and `B2 76 03` (pulse).
3. In the Programmer's Reference Guide (v2.0/v3.0), check the DAW-mode button table for
   `0x76` and for which LEDs are listed as monochrome. Check the 37 user guide's
   hardware overview for the physical Loop label.
