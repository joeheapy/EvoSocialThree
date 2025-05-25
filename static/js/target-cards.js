/**
 * Toggle active state for target outcome cards
 * @param {HTMLElement} card - The clicked target card element
 */
function toggleTargetCard(card) {
  // Remove active class from all target cards
  const allCards = document.querySelectorAll('.target-card')
  allCards.forEach((c) => c.classList.remove('active'))

  // Add active class to clicked card
  card.classList.add('active')

  // Optional: Add analytics or other tracking here
  const targetId = card.getAttribute('data-target-id')
  console.log(`Target card ${targetId} activated`)
}

/**
 * Clear all active states from target cards
 */
function clearActiveTargetCards() {
  const allCards = document.querySelectorAll('.target-card')
  allCards.forEach((card) => card.classList.remove('active'))
}

// Optional: Add keyboard support for accessibility
document.addEventListener('DOMContentLoaded', function () {
  const targetCards = document.querySelectorAll('.target-card')

  targetCards.forEach((card) => {
    // Make cards focusable
    card.setAttribute('tabindex', '0')

    // Add keyboard support
    card.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        toggleTargetCard(this)
      }
    })
  })
})
