import { Route, Routes } from "react-router-dom"
import {Navigation2,  Footer} from "common"


// import image from '../images/fashion.png'

import { Counter, Schedule } from "../../../api"
import { FashionContainer, IrisContainer, LoginContainer, MnistContainer, NaverMovieContainer, SamsungReportContainer, SignUpContainer, StrokeContainer,  } from "exrc"


const Home = () => {
    return (<>
    <table style={{ width: "1200px", height: "550px", margin: "0 auto", border: "1px solid black"}}>
        <thead>
            <tr columns="4" >
                <td style={{ width: "100%", border: "1px solid black"}}>
                    <Navigation2/>
                </td>
            </tr>
        </thead>
        <tbody>
        <tr style={{ width: "20%",height: "80%",  border: "1px solid black"}}>
            <td style={{ width: "100%", border: "1px solid black"}}>
            <Routes>
                <Route path="/counter" element={<Counter/>}></Route>
                <Route path="/todos" element={<Schedule/>}></Route>
                <Route path="/login" element={<LoginContainer/>}></Route>
                <Route path="/signup" element={<SignUpContainer/>}></Route>
                <Route path="/stroke" element={<StrokeContainer/>}></Route>
                <Route path="/iris" element={<IrisContainer/>}></Route>
                <Route path="/fashion" element={<FashionContainer/>}></Route>
                <Route path="/mnist" element={<MnistContainer/>}></Route>
                <Route path="/crawler" element={<NaverMovieContainer/>}></Route>
                <Route path="/samsung-report" element={<SamsungReportContainer/>}></Route>
            </Routes>
            </td>
        </tr>
        {/* <tr>
            <img src={image} alt="Avatar" class="avatar"  />
        </tr> */}
        <tr style={{ width: "100%", height: "20%", border: "1px solid black"}}>
            <td style={{ width: "100%", border: "1px solid black"}}>
                <Footer/>
            </td>
        </tr>
        </tbody>
    </table>
    </>)
}
export default Home