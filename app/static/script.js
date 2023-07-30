function changeImage() {
            var image = document.getElementById("myImage");
            var button = document.getElementById("changeBtn");
            var currentSrc = image.getAttribute("src");
            var currentAlt = image.getAttribute("alt");

            if (currentSrc.includes("before.jpg")) {
                image.setAttribute("src", "{{ url_for('static', filename='images/after.jpg') }}");
                image.setAttribute("alt", "After");
                button.textContent = "See Before";
            } else {
                image.setAttribute("src", "{{ url_for('static', filename='images/before.jpg') }}");
                image.setAttribute("alt", "Before");
                button.textContent = "See After";
            }
        }