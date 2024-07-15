export const helpButton = {
    createButton() {
        const button = document.createElement("button");
        button.innerHTML = `
            <svg t="1719406133878" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2525" width="46" height="46">
                <path d="M502.8 699.1c-19.4 0-35.2-15.7-35.2-35.2v-26.8c-1.1-23.8 2.1-44.9 9.8-63.3 7.6-18.4 22.8-35 45.5-49.7h-0.9c19.8-15.3 36.5-28.9 50.1-40.8 13.6-11.9 23.5-22.1 29.7-30.6 11.3-14.7 17.6-34.2 18.7-58.6 0.6-14.2-1.9-27-7.2-38.7-5.2-11.3-12.7-22.4-22.5-33.4-0.6-0.6-1.2-1.3-1.8-1.9-20.8-20.3-47.4-30.5-79.8-30.5-31.1 0-56.6 12.3-76.4 37-15.3 19-25.5 46.1-30.7 81.4-2.7 18.7-19.9 31.8-38.6 29.7-19.4-2.2-33.4-19.7-31-39.1 2.4-19.4 6.5-37.3 12.4-53.8 9.3-26 21.9-48 37.8-65.8 15.9-17.8 34.6-31.4 56.1-40.8 21.5-9.3 45-14 70.5-14 53.8 0 97.4 15.9 130.8 47.6 34.5 32.8 51.8 73.6 51.8 122.3 0 18.1-2.4 35-7.2 50.5-4.8 15.6-11.2 29.6-19.1 42-8.5 13-21 27.2-37.4 42.5s-36.8 31.4-61.1 48.4c-11.3 7.9-18.7 15.7-22.1 23.4-3.4 7.6-5.1 19.4-5.1 35.3v27.7c0 19.4-15.7 35.2-35.2 35.2h-1.9z" fill="#fffd01" p-id="2526"></path>
                <path d="M504 772m-36 0a36 36 0 1 0 72 0 36 36 0 1 0-72 0Z" fill="#fffd01" p-id="2527"></path>
            </svg>`;
        button.style.backgroundSize = "cover"; 
        button.style.backgroundRepeat = "no-repeat";
        button.style.backgroundPosition = "center";
        button.style.padding = "0px"; 
        button.style.borderRadius = "5px";
        button.style.margin = "5px";
        button.style.backgroundColor = "#007bff00"; 
        button.style.color = "white"; 
        button.style.border = "none";
        button.style.cursor = "pointer";
        button.addEventListener("click", () => {
            this.openHelpWindow();
        });
        return button;
    }, 
    openHelpWindow() {
        const helpWindow = window.open("https://comfly.chat", "_blank", "width=800,height=600,resizable=yes,menubar=no,location=no,status=no");
        
        // Remove default window border and margin
        helpWindow.document.body.style.margin = "0";
        helpWindow.document.body.style.padding = "0";
        helpWindow.document.body.style.overflow = "hidden";
    }
};