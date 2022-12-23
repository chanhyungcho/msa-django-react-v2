import dlearnService from "../api"
import { useState } from "react"

const Iris = ()=> {
    const [inputs, setInputs] = useState({})
    const {SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm} = inputs;
    const onChange = e => {
        e.preventDefault()
        const {value, name} = e.target
        setInputs({...inputs, [name]: value})
    }
    const onClick = e =>{
        e.preventDefault()
        const request = {SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm}
        dlearnService.iris(request)
        let arr = document.getElementsByClassName('box')
        for(let i=0; i<arr.length; i++) arr[i].value = ""
    }
    return (<>
    <form method="post">
        꽃잎 폭 : <input type="text" className="box" name="PetalWidthCm" onChange={onChange} /><br/>
        꽃잎 길이 : <input type="text" className="box" name="PetalLengthCm" onChange={onChange} /><br/>
        꽃받침 폭 : <input type="text" className="box" name="SepalWidthCm" onChange={onChange} /><br/>
        꽃받침 길이 : <input type="text" className="box" name="SepalLengthCm" onChange={onChange} />
        <button onClick={onClick}> 실행 </button>
    </form>
    </>
    )
}
export default Iris