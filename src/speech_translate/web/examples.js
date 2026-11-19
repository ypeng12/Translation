const choice = document.getElementById('example-choice');
const player = document.getElementById('example-player');

function loadSelectedExample() {
  const option = choice.selectedOptions[0];
  player.pause();
  player.src = option.value;
  player.setAttribute('aria-label', 'Play ' + option.textContent);
  document.getElementById('example-reference').textContent = option.dataset.reference;
  document.getElementById('example-source').textContent = 'LibriSpeech clip ' + option.dataset.clip;
}

choice.addEventListener('change', () => {
  loadSelectedExample();
  document.getElementById('clear').click();
});
loadSelectedExample();
