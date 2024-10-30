select  count(*) from badges as b, 		users as u where b.UserId= u.Id  AND u.DownVotes<=1;
