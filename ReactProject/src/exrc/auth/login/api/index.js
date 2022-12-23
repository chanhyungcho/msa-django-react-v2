import axios from 'axios'
const server = `http://localhost:8000`
export const apiLogin = req => axios.post(`${server}/exrc/auth/login`, req)