/**
 * ConfOpt Documentation - Simplified Layout Manager
 * Minimal JavaScript for enhanced UX without breaking RTD functionality
 */

(function() {
  'use strict';

  // Simple debounce utility
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Enhance search input with better UX
  function enhanceSearchInput() {
    const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]');
    if (!searchInput) return;

    // Add placeholder text if not already set
    if (!searchInput.placeholder) {
      searchInput.placeholder = 'Search documentation...';
    }

    // Add smooth focus/blur animations
    searchInput.addEventListener('focus', function() {
      this.parentElement.classList.add('search-focused');
    });

    searchInput.addEventListener('blur', function() {
      this.parentElement.classList.remove('search-focused');
    });
  }

  // Add smooth scroll behavior for navigation links
  function enhanceNavigation() {
    const navLinks = document.querySelectorAll('.wy-menu-vertical a[href^="#"]');

    navLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        const target = document.querySelector(href);

        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });

          // Update URL without jumping
          history.pushState(null, null, href);
        }
      });
    });
  }

  // Add copy button functionality for code blocks (if sphinx-copybutton is not available)
  function addCopyButtons() {
    // Only add if sphinx-copybutton is not already present
    if (document.querySelector('.copybtn')) return;

    const codeBlocks = document.querySelectorAll('.highlight pre');

    codeBlocks.forEach(block => {
      const button = document.createElement('button');
      button.className = 'copy-btn';
      button.innerHTML = '📋';
      button.title = 'Copy to clipboard';
      button.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background: var(--primary-600);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        cursor: pointer;
        opacity: 0.7;
        transition: opacity 0.2s;
      `;

      button.addEventListener('click', async function() {
        try {
          await navigator.clipboard.writeText(block.textContent);
          button.innerHTML = '✅';
          button.title = 'Copied!';
          setTimeout(() => {
            button.innerHTML = '📋';
            button.title = 'Copy to clipboard';
          }, 2000);
        } catch (err) {
          console.warn('Could not copy text: ', err);
        }
      });

      button.addEventListener('mouseenter', function() {
        this.style.opacity = '1';
      });

      button.addEventListener('mouseleave', function() {
        this.style.opacity = '0.7';
      });

      // Add button to code block container
      const container = block.parentElement;
      container.style.position = 'relative';
      container.appendChild(button);
    });
  }

  // Add keyboard navigation enhancement
  function enhanceKeyboardNavigation() {
    document.addEventListener('keydown', function(e) {
      // Alt + S to focus search
      if (e.altKey && e.key === 's') {
        e.preventDefault();
        const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]');
        if (searchInput) {
          searchInput.focus();
          searchInput.select();
        }
      }

      // Escape to blur search
      if (e.key === 'Escape') {
        const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]:focus');
        if (searchInput) {
          searchInput.blur();
        }
      }
    });
  }

  // Add loading state for better perceived performance
  function addLoadingStates() {
    // Add loading class to body initially
    document.body.classList.add('loading');

    // Remove loading class when everything is ready
    window.addEventListener('load', function() {
      setTimeout(() => {
        document.body.classList.remove('loading');
        document.body.classList.add('loaded');
      }, 100);
    });
  }

  // Main initialization function
  function init() {
    try {
      enhanceSearchInput();
      enhanceNavigation();
      addCopyButtons();
      enhanceKeyboardNavigation();
      addLoadingStates();
    } catch (error) {
      console.warn('ConfOpt Layout Manager: Some enhancements failed to initialize:', error);
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Handle page changes for single-page applications
  window.addEventListener('popstate', debounce(init, 100));

  // Export for debugging (optional)
  if (typeof window !== 'undefined') {
    window.ConfOptLayoutManager = {
      init,
      enhanceSearchInput,
      enhanceNavigation,
      addCopyButtons
    };
  }

})();
