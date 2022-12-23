import '../styles/Login.css'
import { useState } from 'react';
import { apiLogin } from '../api';
 

const Login = () => {
    const [inputs, setInputs] = useState({})
    const {email, password} = inputs

    const onChange = e => {
      e.preventDefault()
      const {value, name} = e.target
      setInputs({...inputs, [name]: value})
    }
    const onClick = e => {
      e.preventDefault()
      const request = {email, password}
      alert(`사용자 이름: ${JSON.stringify(request)}`)
      apiLogin(request)
      .then((res)=>{
          console.log(`response is ${res.config.data}`)
          localStorage.setItem('token', JSON.stringify(res.config.data))
      })
      .catch((err) => {
          console.log(err)
          alert('이메일과 비밀번호를 다시 입력해주세요.')
      })
    }
    return (<>
    
    <h1>로그인</h1>
    <p>로그인을 위해 아이디와 비밀번호를 입력해주세요.</p>

    <b>아이디/비밀번호</b>
    <input type="text" placeholder="이메일 입력" name="email" onChange={onChange}/>

    <b>비밀번호</b>
    <input type="password" placeholder="비밀번호 입력" name="password" onChange={onChange}/>
        
    <button onClick={onClick}>로그인</button>
    <button type="button">취소</button><br/>
    <span><h5><a href="#">비밀번호</a>를 잊으셨나요?</h5></span><br/>
    </>)
}

export default Login