import React from 'react';
import { useDispatch } from "react-redux";
import { deleteTodo } from "api/todos/reducers/todo.reducer.js";

const Todo = ({ id, text }) => {

	const dispatch = useDispatch();

	const onClick=()=>{
		dispatch(
			deleteTodo({
				id: id
			})
		)
	}

	return (
		<li className="task-item">
			<div>
				{text}
			</div>
			<div>
				<button className="remove-task-button" onClick={onClick}>Delete</button>
			</div>
		</li>
	);
};

export default Todo;