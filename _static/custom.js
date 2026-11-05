document.addEventListener('DOMContentLoaded', function() {
    var wyNavContent = document.querySelector('.wy-nav-content');

    // Function to hide specific elements within `.wy-nav-content .rst-content`
    function hideElements() {
        if (wyNavContent) {
            // Hide elements within `.rst-content`
            var rstContent = wyNavContent.querySelector('.rst-content');
            if (rstContent) {
                // Hide footer within `.rst-content`
                var footer = rstContent.querySelector('footer');
                if (footer) {
                    footer.style.display = "none";
                }
		// Hide div with role "navigation" within `.rst-content`
                var navigationDiv = rstContent.querySelector('div[role="navigation"]');
                if (navigationDiv) {
                    navigationDiv.style.display = "none";
                }
		// Remove h1 header from div with role "main" within `.rst-content`
                var mainDiv = rstContent.querySelector('div[role="main"]');
                if (mainDiv) {
                    var header = mainDiv.querySelector('h1');
                    if (header) {
                        header.remove();
                    }
                }
            }
        }
    }

    // Observe the cosmoslider section
    var target = document.getElementById('cosmoslider-iframe');
    if (target) {
	wyNavContent.style.maxWidth = '100%';
	wyNavContent.style.height = '100vh';
	// Call the function to hide elements when the DOM is loaded
	hideElements();
    } else {
	wyNavContent.style.maxWidth = '';
	wyNavContent.style.height = '';
    }
});
