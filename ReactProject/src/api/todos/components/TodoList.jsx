import 'api/todos/styles/TodoList.css'
import { useSelector } from 'react-redux'
import { Todo } from '../..'

const TodoList = () => {
    const todos = useSelector((state) => state.todos)
    return (<>
    <ul>
        {todos.map( todo => (
            <Todo key={todo.id} title={todo.text} complete={todo.complete}/>
        ))} 
    </ul>
    </>)
}
export default TodoList