import axios from 'axios'
import { server } from 'context'

export const apiSignUp = req => axios.post(`${server}exrc/auth/signup`, req)