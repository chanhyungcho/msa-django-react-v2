import { combineReducers} from "@reduxjs/toolkit"
import todoReducer from "api/todos/reducers/todo.reducer"

export default combineReducers({
    todos: todoReducer
})

