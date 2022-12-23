import axios from 'axios'
import { server} from 'context'

 export const apiStroke = req => axios.post(`${server}exrc/stroke/stroke`, req)