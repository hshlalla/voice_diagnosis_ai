/*!
* Start Bootstrap - Stylish Portfolio v6.0.6 (https://startbootstrap.com/theme/stylish-portfolio)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-stylish-portfolio/blob/master/LICENSE)
*/
window.addEventListener('DOMContentLoaded', event => {

    const sidebarWrapper = document.getElementById('sidebar-wrapper');
    let scrollToTopVisible = false;
    
    // Closes the sidebar menu
    const menuToggle = document.body.querySelector('.menu-toggle');
    menuToggle.addEventListener('click', event => {
        event.preventDefault();
        sidebarWrapper.classList.toggle('active');
        _toggleMenuIcon();
        menuToggle.classList.toggle('active');
    });

    // Closes responsive menu when a scroll trigger link is clicked
    var scrollTriggerList = [].slice.call(document.querySelectorAll('#sidebar-wrapper .js-scroll-trigger'));
    scrollTriggerList.map(scrollTrigger => {
        scrollTrigger.addEventListener('click', () => {
            sidebarWrapper.classList.remove('active');
            menuToggle.classList.remove('active');
            _toggleMenuIcon();
        })
    });

    function _toggleMenuIcon() {
        const menuToggleBars = document.body.querySelector('.menu-toggle > .fa-bars');
        const menuToggleTimes = document.body.querySelector('.menu-toggle > .fa-xmark');
        if (menuToggleBars) {
            menuToggleBars.classList.remove('fa-bars');
            menuToggleBars.classList.add('fa-xmark');
        }
        if (menuToggleTimes) {
            menuToggleTimes.classList.remove('fa-xmark');
            menuToggleTimes.classList.add('fa-bars');
        }
    }

    // Scroll to top button appear
    document.addEventListener('scroll', () => {
        const scrollToTop = document.body.querySelector('.scroll-to-top');
        if (document.documentElement.scrollTop > 100) {
            if (!scrollToTopVisible) {
                fadeIn(scrollToTop);
                scrollToTopVisible = true;
            }
        } else {
            if (scrollToTopVisible) {
                fadeOut(scrollToTop);
                scrollToTopVisible = false;
            }
        }
    });

    // Popup 기능 추가
    document.getElementById("startButton").addEventListener("click", openPopup);
    document.getElementById("closePopup").addEventListener("click", closePopup);
});

// 팝업 열기
function openPopup() {
    document.getElementById("popup").style.display = "block";
}

// 팝업 닫기
function closePopup() {
    document.getElementById("popup").style.display = "none";
}

// 파일 선택 핸들러 (미리보기 표시)
function handleFiles(input) {
    const fileList = input.files;
    const filePreview = document.getElementById('filePreview');
    filePreview.innerHTML = '';

    if (fileList.length > 11) {
        alert("You can upload up to 11 files.");
        input.value = "";
        return;
    }

    for (let i = 0; i < fileList.length; i++) {
        const listItem = document.createElement("li");
        listItem.textContent = fileList[i].name;
        filePreview.appendChild(listItem);
    }
}

// 파일 업로드 및 서버 전송
async function uploadFiles() {
    const input = document.getElementById("fileInput");
    const fileList = input.files;

    if (fileList.length !== 11) {
        alert("Please upload exactly 11 files.");
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
        formData.append("fileList", fileList[i]);
    }

    try {
        const response = await fetch("/diagnosis", {
            method: "POST",
            body: formData
        });

        const result = await response.json();
        document.getElementById("result").innerText = JSON.stringify(result, null, 2);
    } catch (error) {
        console.error("Error:", error);
        alert("An error occurred while uploading.");
    }
}

// Fade out function
function fadeOut(el) {
    el.style.opacity = 1;
    (function fade() {
        if ((el.style.opacity -= .1) < 0) {
            el.style.display = "none";
        } else {
            requestAnimationFrame(fade);
        }
    })();
};

// Fade in function
function fadeIn(el, display) {
    el.style.opacity = 0;
    el.style.display = display || "block";
    (function fade() {
        var val = parseFloat(el.style.opacity);
        if (!((val += .1) > 1)) {
            el.style.opacity = val;
            requestAnimationFrame(fade);
        }
    })();
};
