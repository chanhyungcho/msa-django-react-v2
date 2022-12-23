import { createSlice } from '@reduxjs/toolkit'

export const Filters = {
  SHOW_ALL: 'SHOW_ALL',
  SHOW_COMPLETED: 'SHOW_COMPLETED',
  SHOW_ACTIVE: 'SHOW_ACTIVE'
}

const filterSlice = createSlice({
  name: 'Filters',
  initialState: Filters.SHOW_ALL,
  reducers: {
    setFilter(state, action) {
      return action.payload
    }
  }
})

export const { setFilter } = filterSlice.actions

export default filterSlice.reducer