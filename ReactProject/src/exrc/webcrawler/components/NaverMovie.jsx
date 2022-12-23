import { useState } from "react"
import webcrawlerService  from "../api"
const NaverCrawler = () => {
    const [movie, setMovie] = useState()

    const onClick = e => {
        e.preventDefault()
        webcrawlerService.apiCrawler().then(res => {
            const json = JSON.parse(res) 
            setMovie(json['result']) //set으로해서 movie에 들어감
        })
        let arr = document.getElementsByClassName('box')
        for(let i=0; i<arr.length; i++) arr[i].value = ""
        
    }
 
    return (<>
    <h2>네이버 영화 크롤러</h2>
    <button onClick ={onClick}>네이버 영화 크롤링</button>
    <p>버튼을 클릭하시면, 네이버 영화 목록이 출력됩니다.</p>
    <table>
        <thead>
            <tr>
                <th>순위</th><th>영화 목록</th>
            </tr>
        </thead>
        <tbody>
        {movie && movie.map(({rank,title})=>(
            <tr key = {rank}><td>{rank}</td><td>{title}</td></tr>
            ))}
        </tbody>
    </table> 
    </>)
}
export default NaverCrawler