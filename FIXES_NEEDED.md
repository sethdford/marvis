# Critical Fixes Needed for Marvis MLX Generator

## Priority 1: Audio Quality (URGENT)

### 1. Add Top-K Sampling
- [ ] Implement `sample_topk()` function in MLX
- [ ] Use `topk=50` as default
- [ ] Apply top-k filtering before categorical sampling

### 2. Fix Temperature
- [ ] Change from `0.7` to `0.9` (original default)

### 3. Set Mimi Codebooks
- [ ] Call `mimi.set_num_codebooks(32)` after loading

### 4. Add Decoder Cache Reset
- [ ] Implement proper cache reset between frames
- [ ] Or ensure decoder starts fresh each frame

## Priority 2: Model Correctness

### 5. Implement Cache System
- [ ] Add `setup_caches()` to MLX model
- [ ] Add `reset_caches()` before generation
- [ ] Use KV-cache for efficient inference

### 6. Add Position Tracking
- [ ] Track `curr_pos` during generation
- [ ] Pass positions to backbone/decoder

### 7. Fix EOS Detection
- [ ] Check if ALL codebooks are 0 (not just c0 >= 2048)
- [ ] `if mx.all(frame_codes == 0): break`

## Priority 3: Features

### 8. Add Context Support
- [ ] Accept `context: List[Segment]`
- [ ] Tokenize and prepend context before generation
- [ ] Enables voice consistency across conversation

### 9. Add Voice Match
- [ ] Support `voice_match=bool` parameter
- [ ] Clone voice characteristics from context

### 10. Increase Chunk Size
- [ ] Change from `4` to `8` frames per chunk
- [ ] Reduces decoding frequency

## Priority 4: Nice-to-Have

### 11. Add Progress Indication
- [ ] Show generation progress
- [ ] Useful for debugging

### 12. Better Error Handling
- [ ] Validate inputs
- [ ] Handle edge cases gracefully

## Current Issues Likely Caused By:

1. **Reverb/Echo** → Missing decoder cache reset + wrong chunk decoding
2. **Poor Quality** → No top-k sampling + wrong temperature
3. **Inconsistent Voice** → No context support + no cache management
4. **Slow/Inefficient** → No KV-cache system

## Estimated Impact:

- **Top-K + Temperature**: +40% quality improvement
- **Mimi set_num_codebooks**: +20% stability
- **Cache Management**: +30% quality, 2x speed
- **Context Support**: Voice consistency across turns

