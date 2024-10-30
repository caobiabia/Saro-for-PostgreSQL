select  count(*) from badges as b, 		users as u where b.UserId= u.Id  AND u.CreationDate<='2014-08-20 08:37:38'::timestamp;
