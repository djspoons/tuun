---
type: research
status: closed
assignee:
blocked_by: []
---

# Which Launchkey control should carry the Loop prefix, and what can we do with it?

## Question

The plan is to use a Launchkey button as the Loop prefix key. Which physical control should it be on the Launchkey models tuun supports (see `src/lib/launchkey.rs`)? What does it send in device DAW mode (CC or note number, channel, port)? Does it send both press and release? Can its LED be set, and with what colours or brightness? Is it already used by the device firmware in a way that conflicts with DAW mode? And how many pads, in which rows, could display and set a Loop value while the prefix is pending?

## Resolution

Findings: [assets/01-launchkey-loop-button.md](../assets/01-launchkey-loop-button.md). Novation's documents could not be fetched directly, so they were cross-checked against tuun's code, Ardour's MK4 surface and a third-party integration.

- tuun supports only the Launchkey MK4 37, which has no button labelled "Repeat". The closest fit is the transport button labelled **Loop**. The other candidate is the **Function** button (CC 105, RGB), which cycles the app-wide Loop today.
- Loop sends CC 118 on channel 1 on the DAW port, with a value of 127 when pressed. tuun ignores it today. Hypothesis (unverified): release is value 0.
- Loop's LED is very likely single-colour, so it is settable by brightness only (a CC on channel 4, as tuun already does for Record).
- No firmware conflict is documented for Loop in DAW mode. Shift plus a pad is reserved by the firmware, so the prefix must not be a Shift chord on the pads.
- There are 16 pads (top row notes 96–103, bottom row 112–119), each settable to a full RGB colour. Only presses reach tuun. The pad arrows (CC 106/107) and `>` (CC 104) are free for paging or confirming.
- To check on the hardware: press Loop and read the log; try `B3 76 7F` / `B3 76 00` on its LED.
