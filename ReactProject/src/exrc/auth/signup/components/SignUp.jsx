import '../styles/SignUp.css'

const SignUp = () =>{

  const onClick = e => {
    e.preventDefault()
  }

  return (<>
    <h2>회원가입</h2>
    <button onClick ={onClick}>회원 등록</button>
    <p>버튼을 클릭하시면, 더미사용자 100명이 등록됩니다.</p>
    
    </>)
}
  
  export default SignUp