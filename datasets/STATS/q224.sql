select  count(*) from votes as v,          badges as b,         users as u where u.Id = v.UserId 	and v.UserId = b.UserId  AND u.Reputation<=371  AND u.UpVotes>=0  AND u.UpVotes<=106  AND v.CreationDate>='2009-02-02 00:00:00'::timestamp;