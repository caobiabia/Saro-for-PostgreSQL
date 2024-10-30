select  count(*) from badges as b, 		users as u where b.UserId= u.Id  AND b.Date>='2010-07-20 03:54:08'::timestamp  AND u.Reputation<=270;
