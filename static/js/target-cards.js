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

  // Enable the Select System Objective button
  updateSelectObjectiveButton()

  // Optional: Add analytics or other tracking here
  const targetId = card.getAttribute('data-target-id')
  console.log(`Target card ${targetId} activated`)
}

/**
 * Update the Select System Objective button state
 */
function updateSelectObjectiveButton() {
  const selectButton = document.getElementById('selectObjectiveButton')
  const activeCard = document.querySelector('.target-card.active')

  if (selectButton) {
    if (activeCard) {
      selectButton.disabled = false
      selectButton.classList.remove('disabled')
    } else {
      selectButton.disabled = true
      selectButton.classList.add('disabled')
    }
  }
}

/**
 * Handle the Select System Objective button click
 */
function selectSystemObjective() {
  const activeCard = document.querySelector('.target-card.active')
  if (!activeCard) {
    alert('Please select a target first by clicking on one of the cards above.')
    return
  }

  const targetId = activeCard.dataset.targetId

  // Create a form and submit it
  const form = document.createElement('form')
  form.method = 'POST'
  form.action = '/select_objective'

  const input = document.createElement('input')
  input.type = 'hidden'
  input.name = 'objective_index'
  input.value = targetId

  form.appendChild(input)
  document.body.appendChild(form)
  form.submit()
}

/**
 * Clear all active states from target cards
 */
function clearActiveTargetCards() {
  const allCards = document.querySelectorAll('.target-card')
  allCards.forEach((card) => card.classList.remove('active'))
  updateSelectObjectiveButton()
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

  // Initialize button state
  updateSelectObjectiveButton()
})
