import { useState } from "react"
import nlpService from "../api"
const SamsungReport = () => {
    const [report, setReport] = useState()

    const onClick = e => {
        e.preventDefault()
        nlpService.apiSamsungReport().then(res => {
            const json = JSON.parse(res) 
            setReport(json['result']) //set으로해서 movie에 들어감
        })
        let arr = document.getElementsByClassName('box')
        for(let i=0; i<arr.length; i++) arr[i].value = ""
        
    }
    return (<>
    <h2>워드클라우드</h2>
    <form method='get'>

          <button onClick={onClick}>Click</button>
        </form>
    <p>버튼을 클릭하시면, 삼성리포트의 가장 많은 빈도의 단어가 출력됩니다.</p>
    <table className='type1'>
        <thead>
            <tr>
                <th>순위</th><th>목록</th>
            </tr>
        </thead>
        <tbody>
        {report && report.map(({rank,title})=>(
            <tr key = {rank}><td>{rank}</td><td>{title}</td></tr>
            ))}
        
        </tbody>
    </table> 
        </>) 
}

export default SamsungReport