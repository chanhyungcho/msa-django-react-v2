import { apiStroke } from "../api"


const Stroke = () => {
    
    const onClick = e => {
        e.preventDefault()
        apiStroke()
        .then(() => {
            alert('Stroke 연결 성공')
        })
    }
    return(<>
    <button onClick={onClick}>가보장</button>
    </>)   
}

export default Stroke