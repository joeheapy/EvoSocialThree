const SpinnerManager = {
  show: function () {
    document.getElementById('loadingSpinner').style.display = 'flex'
  },

  hide: function () {
    document.getElementById('loadingSpinner').style.display = 'none'
  },

  // Initialize event listeners
  init: function () {
    // Add spinner behavior to all forms with class 'spinner-form'
    document.querySelectorAll('form.spinner-form').forEach((form) => {
      form.addEventListener('submit', function () {
        SpinnerManager.show()
      })
    })
  },
}

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', SpinnerManager.init)
