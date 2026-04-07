document.addEventListener("DOMContentLoaded", function() {
    const audio = document.getElementById("bg-music");

    document.addEventListener("visibilitychange", function() {
        if (document.hidden) {
            audio.pause();
        } else {
            audio.play();
        }
    });

    document.addEventListener("click", function startAudio() {
        audio.play();
        document.removeEventListener("click", startAudio);
    }, { once: true });
});