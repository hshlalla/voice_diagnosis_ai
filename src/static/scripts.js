window.addEventListener('DOMContentLoaded', event => {
    const sidebarWrapper = document.getElementById('sidebar-wrapper');
    let scrollToTopVisible = false;

    // 사이드바 메뉴 토글
    const menuToggle = document.body.querySelector('.menu-toggle');
    menuToggle.addEventListener('click', event => {
        event.preventDefault();
        sidebarWrapper.classList.toggle('active');
        _toggleMenuIcon();
        menuToggle.classList.toggle('active');
    });

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

    // 팝업 기능 추가
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

    if (fileList.length !== 11) {
        alert("Please upload exactly 11 files.");
        input.value = "";
        return;
    }

    for (let i = 0; i < fileList.length; i++) {
        const listItem = document.createElement("li");
        listItem.textContent = fileList[i].name;
        filePreview.appendChild(listItem);
    }
}
async function uploadFiles() {
    const input = document.getElementById("fileInput");
    const fileList = input.files;

    // 🚨 파일 개수 검증
    if (!fileList || fileList.length !== 11) {
        alert("Please upload exactly 11 files.");
        return;
    }

    const formData = new FormData();
    for (let i = 0; i < fileList.length; i++) {
        formData.append("fileList", fileList[i]);
    }

    // ⏳ 진행 중 표시 (버튼 비활성화 & 로딩 텍스트 추가)
    const submitButton = document.querySelector("button[onclick='uploadFiles()']");
    submitButton.disabled = true;
    submitButton.innerText = "Uploading...";

    try {
        // 🛰 서버 요청
        const response = await fetch("/diagnosis", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }

        const responseData = await response.json();  // 서버 응답을 JSON으로 변환
        console.log("Diagnosis Response:", responseData);

        if (responseData.status === "success" && responseData.redirect_url) {
            window.location.href = responseData.redirect_url;  // 🚀 결과 페이지로 이동
        } else {
            throw new Error("Invalid response format from server.");
        }

    } catch (error) {
        console.error("Upload Error:", error);
        alert("An error occurred during processing. Please try again.");
    } finally {
        // ⏹ 완료 후 버튼 활성화
        submitButton.disabled = false;
        submitButton.innerText = "Submit";
    }
}



// Fade in/out functions
function fadeOut(el) {
    el.style.opacity = 1;
    (function fade() {
        if ((el.style.opacity -= .1) < 0) {
            el.style.display = "none";
        } else {
            requestAnimationFrame(fade);
        }
    })();
}

function fadeIn(el, display) {
    el.style.opacity = 0;
    el.style.display = display || "block";
    (function fade() {
        let val = parseFloat(el.style.opacity);
        if (!((val += .1) > 1)) {
            el.style.opacity = val;
            requestAnimationFrame(fade);
        }
    })();
}
