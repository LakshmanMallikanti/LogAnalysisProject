const wrapper = document.querySelector('.wrapper');
const registerLink = document.querySelector('.register-link');
const loginLink = document.querySelector('.login-link');
const btnPopup = document.querySelector('.btnLogin-popup');
const iconClose = document.querySelector('.icon-close');
const Eyecon = document.querySelectorAll('.Eyecon');
const x = document.querySelectorAll('#MyInput');

registerLink.onclick = () => {
    wrapper.classList.add('active');
};

loginLink.onclick = () => {
    wrapper.classList.remove('active');
};

btnPopup.onclick = () => {
    wrapper.classList.add('active-popup');
};

iconClose.onclick = () => {
    wrapper.classList.remove('active-popup');
    wrapper.classList.remove('active');
};

for(let i=0;i<Eyecon.length;i++){
    Eyecon[i].addEventListener('click',()=>{
        if(x[i].type==="password"){
            x[i].type="text"
            Eyecon[i].classList.add('fa-eye-slash')
            Eyecon[i].classList.remove('fa-eye')
        }
        else{
            x[i].type="password"
            Eyecon[i].classList.remove('fa-eye-slash')
            Eyecon[i].classList.add('fa-eye')
        }
    })
}