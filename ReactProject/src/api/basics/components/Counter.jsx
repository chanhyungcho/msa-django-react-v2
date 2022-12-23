import { useState } from "react"

const Counter = () => {
    const[count, setCount] = useState(0) // 상태값을 보관한다. init self와 동일
    return(<>
    <h3>카운터</h3>
    <div>클릭한 횟수: {count}</div>
    <button onClick ={() => {setCount(count + 1)}}>
    +클릭버튼
    </button>
    <button onClick ={() => {setCount(count - 1)}}>
    -클릭버튼
    </button>  
    </>)
}

 export default Counter
