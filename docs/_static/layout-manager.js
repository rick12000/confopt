/**
 * ConfOpt Documentation - Dynamic Layout Manager
 * Handles responsive layout calculations for consistent rendering across environments
 */

(function() {
  'use strict';

  let resizeObserver;
  let rafId;

  function updateHeaderHeight() {
    // Cancel any pending updates
    if (rafId) {
      cancelAnimationFrame(rafId);
    }

    rafId = requestAnimationFrame(() => {
      const header = document.querySelector('.wy-side-nav-search');
      if (!header) return;

      try {
        const rect = header.getBoundingClientRect();
        const actualHeight = Math.max(rect.height, 80); // Minimum 80px
        const maxHeight = Math.min(actualHeight, 200); // Maximum 200px

        document.documentElement.style.setProperty(
          '--dynamic-header-height',
          `${maxHeight}px`
        );

        // Also update the navigation menu positioning
        const menu = document.querySelector('.wy-menu-vertical');
        if (menu) {
          menu.style.top = `${maxHeight}px`;
          menu.style.height = `calc(100vh - ${maxHeight}px)`;
        }

        // Dispatch custom event for other scripts that might need this info
        window.dispatchEvent(new CustomEvent('headerHeightUpdated', {
          detail: { height: maxHeight }
        }));

      } catch (error) {
        console.warn('Layout Manager: Error updating header height:', error);
      }
    });
  }

  function initializeLayoutManager() {
    // Initial update
    updateHeaderHeight();

    // Handle window resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(updateHeaderHeight, 100);
    });

    // Handle orientation change on mobile
    window.addEventListener('orientationchange', () => {
      setTimeout(updateHeaderHeight, 200);
    });

    // Use ResizeObserver for more precise header size tracking
    if (window.ResizeObserver) {
      const header = document.querySelector('.wy-side-nav-search');
      if (header) {
        resizeObserver = new ResizeObserver((entries) => {
          for (const entry of entries) {
            if (entry.target === header) {
              updateHeaderHeight();
              break;
            }
          }
        });
        resizeObserver.observe(header);
      }
    }

    // Handle dynamic content changes
    if (window.MutationObserver) {
      const observer = new MutationObserver((mutations) => {
        let shouldUpdate = false;

        mutations.forEach((mutation) => {
          if (mutation.type === 'childList' || mutation.type === 'attributes') {
            const target = mutation.target;
            if (target.closest && target.closest('.wy-side-nav-search')) {
              shouldUpdate = true;
            }
          }
        });

        if (shouldUpdate) {
          setTimeout(updateHeaderHeight, 50);
        }
      });

      const header = document.querySelector('.wy-side-nav-search');
      if (header) {
        observer.observe(header, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ['style', 'class']
        });
      }
    }
  }

  function handleFontLoad() {
    // Fonts can affect layout, so update when they're loaded
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(updateHeaderHeight);
    }
  }

  function cleanupLayoutManager() {
    if (resizeObserver) {
      resizeObserver.disconnect();
    }
    if (rafId) {
      cancelAnimationFrame(rafId);
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeLayoutManager);
  } else {
    initializeLayoutManager();
  }

  // Handle font loading
  handleFontLoad();

  // Cleanup on page unload
  window.addEventListener('beforeunload', cleanupLayoutManager);

  // Export for debugging
  window.ConfOptLayoutManager = {
    updateHeaderHeight,
    initializeLayoutManager,
    cleanupLayoutManager
  };

})();
