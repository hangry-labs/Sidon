// SPDX-License-Identifier: Apache-2.0
const players = [...document.querySelectorAll('.audio-comparison audio')];
for (const player of players) {
  player.addEventListener('play', () => {
    for (const other of players) if (other !== player) other.pause();
  });
}
