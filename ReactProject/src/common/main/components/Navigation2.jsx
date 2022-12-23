import * as React from 'react';
import Box from '@mui/material/Box';
import BottomNavigation from '@mui/material/BottomNavigation';
// import BottomNavigationAction from '@mui/material/BottomNavigationAction';
// import RestoreIcon from '@mui/icons-material/Restore';
// import FavoriteIcon from '@mui/icons-material/Favorite';
// import LocationOnIcon from '@mui/icons-material/LocationOn';
import { Link } from "react-router-dom"

const Navigation2 =() => {
  const [value, setValue] = React.useState(0);

  return (
    <Box sx={{ width: 500 }}>
      <BottomNavigation
        showlabels
        value={value}
        onChange={(event, newValue) => {
          setValue(newValue);
        }}
      >
        <Link to ='/home' style={{width:60, margin:10}}>Home</Link>
        <Link to = '/counter'style={{width:60, margin:10}}>Counter</Link>
        <Link to = '/todos'style={{width:60, margin:10}}>Todos</Link>
        <Link to = '/signup'style={{width:60, margin:10}}>Sign UP</Link>
        <Link to = '/login'style={{width:60, margin:10}}>Login</Link>
        <Link to = '/stroke'style={{width:60, margin:10}}>Stroke</Link>
        <Link to = '/iris'style={{width:60, margin:10}}>Iris</Link>
        <Link to = '/fashion'style={{width:60, margin:10}}>Fashion</Link>
        <Link to = '/mnist'style={{width:60, margin:10}}>Mnist</Link>
        <Link to = '/crawler'style={{width:60, margin:10}}>Crawling</Link>
        <Link to = '/samsung-report'style={{width:60, margin:10}}>samsung-report</Link>
      </BottomNavigation>
    </Box>
  );
}

export default Navigation2