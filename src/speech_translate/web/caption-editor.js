const element = id => document.getElementById(id);

export function openCaptionEditor(rows, applyCorrection) {
  const dialog = element('edit-dialog');
  const form = element('edit-form');
  const segment = element('edit-segment');
  const input = element('edit-english');
  const message = element('edit-message');
  const save = element('edit-save');
  const request = new AbortController();
  segment.replaceChildren(...rows.map((row, i) => new Option(
    `${i + 1}. ${row.english.slice(0, 65)}`, String(i)
  )));
  segment.hidden = element('edit-segment-label').hidden = rows.length === 1;
  segment.disabled = input.disabled = false;
  function loadSegment() {
    input.value = rows[segment.selectedIndex].english;
    message.textContent = '';
    save.disabled = true;
  }
  segment.onchange = loadSegment;
  input.oninput = () => {
    save.disabled = !input.value.trim() || input.value.trim() === rows[segment.selectedIndex].english;
  };
  element('edit-cancel').onclick = () => dialog.close();
  dialog.onclose = () => request.abort();
  form.onsubmit = async event => {
    event.preventDefault();
    if (save.disabled) return;
    const index = segment.selectedIndex;
    const row = rows[index];
    const english = input.value.trim();
    segment.disabled = input.disabled = save.disabled = true;
    message.textContent = 'Translating your correction…';
    try {
      const response = await fetch('/api/translate/text', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: english, language: row.language, clean: false}),
        signal: request.signal,
      });
      const result = await response.json();
      if (!response.ok || typeof result.translation !== 'string' || !result.translation.trim()) {
        throw new Error(typeof result.detail === 'string' ? result.detail : 'Translation failed. Please try again.');
      }
      if (!dialog.open || request.signal.aborted) return;
      applyCorrection(index, english, result.translation);
      dialog.close();
    } catch (error) {
      if (!request.signal.aborted) {
        message.textContent = error.message;
        segment.disabled = input.disabled = save.disabled = false;
      }
    }
  };
  loadSegment();
  dialog.showModal();
  input.focus();
}
